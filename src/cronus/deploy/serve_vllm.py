"""Production Ultra-Low-Latency HTTP API Server for Cronus-9B powered by vLLM.

Features:
- Full native 262,144-token context window (`max_model_len=262144`, zero silent truncation)
- Chunked prefill capped at 2,048 tokens (`enable_chunked_prefill=True, max_num_batched_tokens=2048`)
- Piecewise CUDA Graphs captured across `1..1024` tokens (`max_cudagraph_capture_size=1024`)
- 64-row FP32 `W0` `lm_head` slicing (`248,320 -> 64`, self-verified `<= 1e-6` probability delta at boot)
- Stage-1 default adapter pre-merged in FP8 128x128 checkpoint (`lora_request=None` for `W0`/`permission`)
  plus standalone residual LoRAs `role_decision_r16` and `role_scope_r8`
- Hybrid `mamba_cache_mode="align"` prefix caching + multi-question continuous batching (`1..64` per call)
"""

from __future__ import annotations

import argparse
import hashlib
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import math
import os
from pathlib import Path
import platform
import string
import threading
import time
import traceback

import torch
from torch import nn

MEM_FRAC = float(os.environ.get("CUDA_MEM_FRACTION", "0.33"))
assert 0.0 < MEM_FRAC <= 0.35, f"CUDA_MEM_FRACTION={MEM_FRAC} violates 0.35 safety ceiling"

import transformers
from transformers import AutoTokenizer
import vllm
from vllm import LLM, SamplingParams
from vllm.lora.request import LoRARequest

from cronus.formatting import qwen_prompt
from cronus.paths import resolve_bundle_layout
from cronus.task_heads import head_for

GPU_MEM_UTIL = float(os.environ.get("CRONUS_GPU_MEM_UTIL", os.environ.get("GUARD_GPU_MEM_UTIL", "0.30")))
assert 0.0 < GPU_MEM_UTIL <= 0.35, f"GPU_MEM_UTIL={GPU_MEM_UTIL} violates 0.35 safety ceiling"

MAX_CONTEXT_TOKENS = int(os.environ.get("MAX_CONTEXT_TOKENS", "262144"))
MAX_BODY_BYTES = int(os.environ.get("MAX_BODY_BYTES", str(16 * 1024 * 1024)))
CHUNK_SIZE = 2048
MAX_NUM_SEQS = 8
CG_SIZES = [1, 2, 4, 8, 16, 32, 64, 96, 128, 160, 192, 256, 320, 384, 448, 512, 640, 768, 896, 1024]


def sha_file(path: Path) -> str:
    """Return SHA-256 hex digest of a file."""
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def token_sha(ids: list[int]) -> str:
    """Deterministic SHA-256 digest of packed input token IDs."""
    raw = ("[" + ",".join(map(str, ids)) + "]").encode("ascii")
    return hashlib.sha256(raw).hexdigest()


def logsumexp_list(xs: list[float]) -> float:
    """Numerically stable log-sum-exp over a non-empty float list."""
    m = max(xs)
    return m + math.log(math.fsum(math.exp(x - m) for x in xs))


def calibrated_probs(logits_list: list[float], temp_mult: float) -> list[float]:
    """Compute temperature-scaled softmax probabilities."""
    scaled = [float(v) / temp_mult for v in logits_list]
    mx = max(scaled)
    exps = [math.exp(v - mx) for v in scaled]
    den = math.fsum(exps)
    return [v / den for v in exps]


def _shrink_lm_head_64(model) -> dict:
    inner = getattr(model, "language_model", model)
    head = inner.lm_head
    base_head = getattr(head, "base_layer", head)
    lp = inner.logits_processor
    base_lp = getattr(lp, "base_layer", lp)
    old_shape = list(base_head.weight.shape)
    w64 = base_head.weight[:64, :].detach().clone().contiguous()
    base_head.weight = nn.Parameter(w64, requires_grad=False)
    for attr in ("num_embeddings", "org_vocab_size", "num_embeddings_per_partition", "vocab_end_index"):
        if hasattr(base_head, attr):
            setattr(base_head, attr, 64)
        if head is not base_head and hasattr(head, attr):
            try:
                setattr(head, attr, 64)
            except AttributeError:
                pass
    for attr in ("vocab_size", "org_vocab_size"):
        if hasattr(base_lp, attr):
            setattr(base_lp, attr, 64)
    if hasattr(lp, "lora_b_stacked") and lp.lora_b_stacked is not None:
        lp.lora_b_stacked = lp.lora_b_stacked[:, :, :64, :].contiguous()
    if hasattr(model, "logits_processor") and getattr(model, "logits_processor") is lp:
        model.logits_processor = base_lp
    inner.lm_head = base_head
    inner.logits_processor = base_lp
    torch.cuda.empty_cache()
    return {"old_shape": old_shape, "new_shape": list(base_head.weight.shape), "unwrapped_lora_lp": lp is not base_lp}


class PromptAdapter:
    """Fast Rust tokenizer + parity-verified Qwen3.5 non-thinking prompt packer."""

    def __init__(self, model_dir: Path, native_context: int = 262144) -> None:
        self.tokenizer = AutoTokenizer.from_pretrained(str(model_dir), local_files_only=True, trust_remote_code=False)
        self.fast_tok = self.tokenizer.backend_tokenizer
        self.native_context = native_context
        self.label_ids: list[int] = []
        for ch in string.ascii_uppercase:
            ids = self.tokenizer.encode(ch, add_special_tokens=False, truncation=False)
            assert len(ids) == 1
            self.label_ids.append(ids[0])
        assert max(self.label_ids) < 64, self.label_ids

        ref_txt = self.tokenizer.apply_chat_template(
            [{"role": "system", "content": "S"}, {"role": "user", "content": "U"}],
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        if ref_txt.endswith("<think>\n"):
            ref_txt += "\n</think>\n\n"
        assert self._native_chat("S", "U") == ref_txt
        assert self.fast_tok.encode(ref_txt, add_special_tokens=False).ids == self.tokenizer.encode(
            ref_txt, add_special_tokens=False, truncation=False
        )

    def _native_chat(self, system: str, user: str) -> str:
        return (
            f"<|im_start|>system\n{system}<|im_end|>\n"
            f"<|im_start|>user\n{user}<|im_end|>\n"
            f"<|im_start|>assistant\n<think>\n\n</think>\n\n"
        )

    def pack_many(self, state, questions: list[dict], max_ctx: int) -> list[dict]:
        """Encode one shared state paired with 1..64 questions."""
        prompts: list[str] = []
        metas: list[tuple[list[str], list[list[int]], float]] = []
        char_cap = max_ctx * 24
        for question in questions:
            prompt, names, groups, temp = qwen_prompt(self, state, question)
            if len(prompt) > char_cap:
                raise OverflowError(
                    f"Prompt character length {len(prompt)} exceeds bound for max context {max_ctx}; silent truncation is forbidden"
                )
            prompts.append(prompt)
            metas.append((names, groups, temp))
        encodings = self.fast_tok.encode_batch(prompts, add_special_tokens=False)
        out: list[dict] = []
        for enc, (names, groups, temp) in zip(encodings, metas):
            ids = enc.ids
            if not ids:
                raise ValueError("Encoded prompt cannot be empty")
            if len(ids) > max_ctx:
                raise OverflowError(
                    f"Input length {len(ids)} exceeds max supported context {max_ctx}; silent truncation is forbidden"
                )
            out.append(
                {
                    "ids": ids,
                    "names": names,
                    "groups": groups,
                    "temperature": temp,
                    "input_ids_sha256": token_sha(ids),
                }
            )
        return out

    def pack(self, state, question: dict, max_ctx: int) -> dict:
        """Encode a single state + question pair."""
        return self.pack_many(state, [question], max_ctx)[0]


class CronusVLLMEngine:
    """Thread-safe vLLM inference engine for Cronus-9B two-role zero-generation routing."""

    def __init__(self, bundle_dir: str | Path | None = None, quant_mode: str | None = None) -> None:
        t_boot = time.perf_counter()
        torch.set_num_threads(8)
        torch.cuda.set_per_process_memory_fraction(MEM_FRAC)

        layout = resolve_bundle_layout(bundle_dir)
        self.layout = layout
        self.quant_mode = (quant_mode or os.environ.get("CRONUS_QUANTIZATION", os.environ.get("GUARD_QUANTIZATION", "fp8"))).strip().lower()

        self.manifest_sha = sha_file(layout.manifest_path)
        expected_manifest = os.environ.get("CRONUS_EXPECTED_MANIFEST_SHA256", "").strip()
        if expected_manifest and self.manifest_sha != expected_manifest:
            raise RuntimeError(f"Manifest SHA-256 mismatch: expected {expected_manifest}, got {self.manifest_sha}")

        self.spec = json.loads(layout.manifest_path.read_text())
        self.cal = json.loads(layout.calibration_path.read_text())
        self.cal_sha = sha_file(layout.calibration_path)
        self.temp_mult = float(self.cal["temperature_multiplier"])
        self.op_threshold = float(self.cal["operating_threshold"])
        assert self.temp_mult >= 1.0 and 0.90 <= self.op_threshold <= 1.0
        assert 1 <= MAX_CONTEXT_TOKENS <= 262144

        self.pa = PromptAdapter(layout.model_dir, native_context=262144)
        self.loras = {
            "decision": LoRARequest("role_decision_r16", 1, str(layout.lora_decision_dir)),
            "scope": LoRARequest("role_scope_r8", 2, str(layout.lora_scope_dir)),
        }
        quant_arg = None if self.quant_mode in ("", "none", "bf16", "null") else self.quant_mode

        self.llm = LLM(
            model=str(layout.model_dir),
            language_model_only=True,
            trust_remote_code=False,
            dtype="bfloat16",
            quantization=quant_arg,
            hf_overrides={"head_dtype": "float32"},
            max_model_len=MAX_CONTEXT_TOKENS,
            gpu_memory_utilization=GPU_MEM_UTIL,
            enable_chunked_prefill=True,
            max_num_batched_tokens=CHUNK_SIZE,
            max_num_seqs=MAX_NUM_SEQS,
            enable_prefix_caching=True,
            mamba_cache_mode="align",
            enable_lora=True,
            max_loras=2,
            max_lora_rank=int(os.environ.get("CRONUS_MAX_LORA_RANK", os.environ.get("GUARD_MAX_LORA_RANK", "16"))),
            max_logprobs=64,
            logprobs_mode="raw_logprobs",
            compilation_config={
                "max_cudagraph_capture_size": 1024,
                "cudagraph_capture_sizes": CG_SIZES,
            },
            seed=7813,
            disable_log_stats=True,
        )

        self.lock = threading.Lock()
        self.started_monotonic = time.monotonic()
        self.total_requests = 0
        self.total_questions = 0
        self.req_seq = 0

        self.lm_head_slice_info, self.lm_head_slice_diff = self._verify_and_install_64row_head()
        self.cold_load_seconds = time.perf_counter() - t_boot

    def _next_internal_id(self) -> str:
        self.req_seq += 1
        return f"g_{self.req_seq}"

    def _verify_and_install_64row_head(self) -> tuple[dict, float]:
        dummy_state = {"tool": "echo", "args": {"text": "warmup"}, "trusted": {"scope": "authorized"}}
        warm_qs = [
            {"type": "noul", "instructions": "Is production harm expected?", "criteria": {"true": "Harm.", "false": "No harm."}},
            {
                "type": "choice",
                "instructions": "Select pre-execution action.",
                "criteria": {"approve": "Allow.", "escalate": "Ask human.", "reject": "Deny."},
            },
            {"type": "noul", "instructions": "Is this tool call out of scope?", "criteria": {"true": "Out of scope.", "false": "In scope."}},
        ]
        before = self._run_packed_batch([self.pa.pack(dummy_state, warm_qs[0], MAX_CONTEXT_TOKENS)], [warm_qs[0]], None)[0]
        self.llm.reset_prefix_cache()
        info = self.llm.apply_model(_shrink_lm_head_64)
        after = self._run_packed_batch([self.pa.pack(dummy_state, warm_qs[0], MAX_CONTEXT_TOKENS)], [warm_qs[0]], None)[0]
        diff = max(abs(a - b) for a, b in zip(before["probabilities"], after["probabilities"]))
        assert diff <= 1e-6, f"64-row lm_head slice parity check failed: diff={diff}"
        for question in warm_qs:
            _ = self._run_packed_batch([self.pa.pack(dummy_state, question, MAX_CONTEXT_TOKENS)], [question], None)
        self.llm.reset_prefix_cache()
        return {"engine_apply": info}, diff

    def _format_question_result(self, packed: dict, question: dict, lp_map: dict, proceed_override) -> dict:
        head = head_for(question)
        active_adapters = ["default"] + ([f"role_{head}"] if head in self.loras else [])
        zs = [
            logsumexp_list([float(lp_map[self.pa.label_ids[idx]].logprob) for idx in grp]) / packed["temperature"]
            for grp in packed["groups"]
        ]
        norm = logsumexp_list(zs)
        ps = [math.exp(z - norm) for z in zs]
        assert all(math.isfinite(v) for v in zs) and all(math.isfinite(v) for v in ps)

        names = packed["names"]
        best_idx = max(range(len(ps)), key=ps.__getitem__)
        argmax_label = names[best_idx]
        cal_ps = calibrated_probs(zs, self.temp_mult) if head == "decision" else ps

        gate_action = argmax_label
        native_tau090_action = argmax_label
        proceed_opt = proceed_override
        if proceed_opt is None and head == "decision" and "approve" in names:
            proceed_opt = "approve"
        if proceed_opt in names:
            p_idx = names.index(proceed_opt)
            cal_best = max(range(len(cal_ps)), key=cal_ps.__getitem__)
            if cal_best == p_idx and cal_ps[p_idx] >= self.op_threshold:
                gate_action = "auto_allow"
            elif argmax_label == "reject":
                gate_action = "reject"
            else:
                gate_action = "escalate_confirm"

            if best_idx == p_idx and ps[p_idx] >= 0.90:
                native_tau090_action = "auto_allow"
            elif argmax_label == "reject":
                native_tau090_action = "reject"
            else:
                native_tau090_action = "escalate_confirm"
        elif head == "scope" and "true" in names:
            gate_action = "block_out_of_scope" if ps[names.index("true")] >= 0.5 else "in_scope"
            native_tau090_action = gate_action

        return {
            "task_head": head,
            "active_adapters": active_adapters,
            "names": names,
            "logits": zs,
            "probabilities": ps,
            "calibrated_probabilities": cal_ps,
            "argmax_label": argmax_label,
            "harness_gate_action": gate_action,
            "native_tau090_action": native_tau090_action,
            "operating_temperature": self.temp_mult,
            "operating_threshold": self.op_threshold,
            "input_tokens": len(packed["ids"]),
            "input_ids_sha256": packed["input_ids_sha256"],
        }

    def _run_packed_batch(self, packed_list: list[dict], questions: list[dict], proceed_override) -> list[dict]:
        outputs_by_idx: dict[int, dict] = {}
        for start in range(0, len(packed_list), MAX_NUM_SEQS):
            sub_x = packed_list[start : start + MAX_NUM_SEQS]
            sub_q = questions[start : start + MAX_NUM_SEQS]
            id_to_local: dict[str, int] = {}
            for local_i, (packed, question) in enumerate(zip(sub_x, sub_q)):
                head = head_for(question)
                lreq = self.loras.get(head, None)
                req_tok_ids = [self.pa.label_ids[idx] for grp in packed["groups"] for idx in grp]
                sp = SamplingParams(
                    temperature=0.0,
                    max_tokens=1,
                    ignore_eos=True,
                    stop_token_ids=[],
                    logprobs=len(req_tok_ids),
                    logprob_token_ids=req_tok_ids,
                    detokenize=False,
                )
                rid = self._next_internal_id()
                id_to_local[rid] = start + local_i
                self.llm.llm_engine.add_request(rid, {"prompt_token_ids": packed["ids"]}, sp, lora_request=lreq)

            finished = 0
            while finished < len(sub_x):
                step_outs = self.llm.llm_engine.step()
                for step_out in step_outs:
                    if step_out.finished and step_out.request_id in id_to_local:
                        g_idx = id_to_local[step_out.request_id]
                        lp_map = step_out.outputs[0].logprobs[0]
                        outputs_by_idx[g_idx] = self._format_question_result(
                            packed_list[g_idx], questions[g_idx], lp_map, proceed_override
                        )
                        finished += 1
        return [outputs_by_idx[i] for i in range(len(packed_list))]

    def clear_sessions(self, session_key=None) -> int:
        """Reset vLLM prefix cache."""
        with self.lock:
            self.llm.reset_prefix_cache()
            return 1

    def health(self) -> dict:
        """Return service readiness and pinned runtime metadata."""
        return {
            "status": "ok",
            "service": "cronus-guard",
            "engine": "vllm",
            "vllm_version": vllm.__version__,
            "quantization": "fp8_128x128_blockwise" if self.quant_mode == "fp8" else "bfloat16",
            "cudagraph_max_size": 1024,
            "lm_head_sliced_vocab": 64,
            "lm_head_slice_parity_max_diff": self.lm_head_slice_diff,
            "model_name": self.spec.get("model_name", "Cronus-9B"),
            "manifest_sha256": self.manifest_sha,
            "calibration_sha256": self.cal_sha,
            "calibrated_temperature_multiplier": self.temp_mult,
            "calibrated_operating_threshold": self.op_threshold,
            "loaded_adapters": ["default", "role_decision", "role_scope"],
            "max_context_tokens": MAX_CONTEXT_TOKENS,
            "native_context_tokens": self.pa.native_context,
            "chunked_prefill_tokens": CHUNK_SIZE,
            "prefix_caching_enabled": True,
            "mamba_cache_mode": "align",
            "total_requests": self.total_requests,
            "total_questions_evaluated": self.total_questions,
            "uptime_seconds": round(time.monotonic() - self.started_monotonic, 2),
            "cold_load_seconds": round(self.cold_load_seconds, 2),
            "gpu": torch.cuda.get_device_name(0),
            "node": platform.node(),
            "torch": torch.__version__,
            "transformers": transformers.__version__,
            "no_generation": True,
            "no_silent_truncation": True,
        }

    def evaluate(self, payload: dict) -> dict:
        """Score one question or a batch of 1..64 questions on a shared state."""
        t_req = time.perf_counter()
        if not isinstance(payload, dict) or "state" not in payload:
            raise ValueError("Request JSON must include 'state' and either 'question' or 'questions'")

        has_single = "question" in payload
        has_multi = "questions" in payload
        if has_single == has_multi:
            raise ValueError("Request JSON must include exactly one of 'question' or 'questions'")

        eff_max_ctx = MAX_CONTEXT_TOKENS
        if "max_context_tokens" in payload and payload["max_context_tokens"] is not None:
            req_cap = int(payload["max_context_tokens"])
            if req_cap < 1 or req_cap > MAX_CONTEXT_TOKENS:
                raise ValueError(f"max_context_tokens must be in [1, {MAX_CONTEXT_TOKENS}]")
            eff_max_ctx = req_cap

        req_id = str(payload.get("id", "req"))
        state = payload["state"]
        proceed_override = payload.get("proceed_option")
        use_cache = bool(payload.get("use_prefix_cache", True))

        questions = [payload["question"]] if has_single else payload["questions"]
        if not isinstance(questions, list) or not questions or len(questions) > 64:
            raise ValueError("'questions' must be a non-empty list of 1..64 question objects")

        packed_list = self.pa.pack_many(state, questions, eff_max_ctx)

        with self.lock:
            if not use_cache:
                self.llm.reset_prefix_cache()
            t_fwd = time.perf_counter()
            results = self._run_packed_batch(packed_list, questions, proceed_override)
            fwd_ms = (time.perf_counter() - t_fwd) * 1000.0
            self.total_requests += 1
            self.total_questions += len(results)

        total_ms = (time.perf_counter() - t_req) * 1000.0
        if has_single:
            res = results[0]
            return {
                "id": req_id,
                "status": "ok",
                **res,
                "prefix_cache_used": use_cache,
                "forward_ms": round(fwd_ms, 3),
                "server_ms": round(total_ms, 3),
            }

        return {
            "id": req_id,
            "status": "ok",
            "question_count": len(results),
            "results": results,
            "prefix_cache_used": use_cache,
            "batch_forward_ms": round(fwd_ms, 3),
            "per_question_forward_ms": round(fwd_ms / len(results), 3),
            "server_ms": round(total_ms, 3),
        }


ENGINE: CronusVLLMEngine | None = None


class GuardHTTPHandler(BaseHTTPRequestHandler):
    """Minimal HTTP/1.1 JSON handler for `/healthz`, `/evaluate`, and `/session/clear`."""

    protocol_version = "HTTP/1.1"
    disable_nagle_algorithm = True

    def log_message(self, fmt_str, *args) -> None:
        pass

    def _send_json(self, code: int, obj: dict) -> None:
        body = json.dumps(obj, ensure_ascii=False, allow_nan=False, separators=(",", ":")).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:
        if self.path.split("?", 1)[0] in ("/healthz", "/health"):
            assert ENGINE is not None
            self._send_json(200, ENGINE.health())
        else:
            self._send_json(404, {"error": "not_found", "path": self.path})

    def do_POST(self) -> None:
        assert ENGINE is not None
        route_path = self.path.split("?", 1)[0]
        try:
            length = int(self.headers.get("Content-Length", "0"))
            if length < 0 or length > MAX_BODY_BYTES:
                self._send_json(413, {"status": "error", "error_type": "PayloadTooLarge", "max_bytes": MAX_BODY_BYTES})
                return
            raw = self.rfile.read(length) if length > 0 else b"{}"
            payload = json.loads(
                raw,
                parse_constant=lambda c: (_ for _ in ()).throw(ValueError(f"Invalid JSON constant {c}")),
            )
            if route_path == "/session/clear":
                cleared = ENGINE.clear_sessions(payload.get("key") if isinstance(payload, dict) else None)
                self._send_json(200, {"status": "cleared", "cleared_entries": cleared})
            elif route_path == "/evaluate":
                res = ENGINE.evaluate(payload)
                self._send_json(200, res)
            else:
                self._send_json(404, {"error": "not_found", "path": self.path})
        except OverflowError as exc:
            self._send_json(413, {"status": "error", "error_type": "ContextLengthExceeded", "message": str(exc)})
        except ValueError as exc:
            self._send_json(400, {"status": "error", "error_type": "BadRequest", "message": str(exc)})
        except Exception as exc:
            self._send_json(
                500,
                {
                    "status": "error",
                    "error_type": type(exc).__name__,
                    "message": str(exc),
                    "traceback": traceback.format_exc(),
                },
            )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle-dir", type=Path, default=None, help="Root directory of Cronus-9B HF bundle or local FP8 bundle.")
    parser.add_argument("--host", default=os.environ.get("CRONUS_HOST", os.environ.get("GUARD_HOST", "0.0.0.0")))
    parser.add_argument("--port", type=int, default=int(os.environ.get("CRONUS_PORT", os.environ.get("GUARD_PORT", "18971"))))
    parser.add_argument("--quantization", default=None, help="Override quantization mode (default fp8).")
    return parser.parse_args()


def main() -> None:
    global ENGINE
    args = parse_args()
    ENGINE = CronusVLLMEngine(bundle_dir=args.bundle_dir, quant_mode=args.quantization)
    server = ThreadingHTTPServer((args.host, args.port), GuardHTTPHandler)
    print(
        json.dumps(
            {
                "event": "cronus_vllm_api_ready",
                "host": args.host,
                "port": args.port,
                "quantization": ENGINE.quant_mode,
                "max_context_tokens": MAX_CONTEXT_TOKENS,
                "manifest_sha256": ENGINE.manifest_sha,
                "calibration_sha256": ENGINE.cal_sha,
                "temperature_multiplier": ENGINE.temp_mult,
                "operating_threshold": ENGINE.op_threshold,
                "lm_head_slice_diff": ENGINE.lm_head_slice_diff,
                "cold_load_seconds": round(ENGINE.cold_load_seconds, 2),
            }
        ),
        flush=True,
    )
    server.serve_forever()


if __name__ == "__main__":
    main()

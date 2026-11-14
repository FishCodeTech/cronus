"""Fallback Native Transformers + FLA HTTP API Server for Static Pre-Merged Cronus-9B.

Serves a static pre-merged HF checkpoint (zero runtime PEFT adapters) with:
- Pinned Triton FLA autotuner configs (`cronus.kernels.native_pin`)
- Pre-allocated KV buffers (`PreallocatedDynamicLayer`) and chunked prefill (`CHUNK_SIZE = 2048`)
- `torch.cuda.synchronize()` rate-limiting inside every chunked prefill loop
- Bounded prefix caching (`OrderedDict`) and full 262,144-token context enforcement
"""

from __future__ import annotations

import argparse
from collections import OrderedDict
import hashlib
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import importlib
import json
import math
import os
from pathlib import Path
import platform
import string
import threading
import time
import traceback

from safetensors.torch import load_file
import torch
from torch.nn.attention.bias import CausalBias, causal_lower_right
import transformers
from transformers import AutoConfig, AutoTokenizer, Qwen3_5ForCausalLM
from transformers.cache_utils import DynamicCache, DynamicLayer
from transformers.masking_utils import ALL_MASK_ATTENTION_FUNCTIONS
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
import triton
from triton.runtime.autotuner import Autotuner

from cronus.formatting import qwen_prompt
from cronus.kernels.native_pin import install
from cronus.paths import default_config_path, resolve_bundle_layout
from cronus.task_heads import head_for

CUDA_MEM_FRACTION = float(os.environ.get("CUDA_MEM_FRACTION", "0.33"))
assert 0.0 < CUDA_MEM_FRACTION <= 0.35, CUDA_MEM_FRACTION

MAX_CONTEXT_TOKENS = int(os.environ.get("MAX_CONTEXT_TOKENS", "262144"))
MAX_BODY_BYTES = int(os.environ.get("MAX_BODY_BYTES", str(32 * 1024 * 1024)))
MAX_PREFIX_CACHE_ENTRIES = int(os.environ.get("MAX_PREFIX_CACHE_ENTRIES", "8"))
CHUNK_SIZE = 2048


def sha_file(path: Path) -> str:
    """Return SHA-256 hex digest of a file."""
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def digest_obj(obj: object) -> str:
    """Return deterministic SHA-256 hex digest of a JSON-serializable object."""
    raw = json.dumps(obj, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    return hashlib.sha256(raw).hexdigest()


def calibrated_probs(logits_list: list[float], temp_mult: float) -> list[float]:
    """Compute temperature-scaled softmax probabilities."""
    scaled = [float(v) / temp_mult for v in logits_list]
    mx = max(scaled)
    exps = [math.exp(v - mx) for v in scaled]
    den = math.fsum(exps)
    return [v / den for v in exps]


def clone_dynamic_cache(src: DynamicCache, text_config) -> DynamicCache:
    """Deep-clone cached KV / recurrent states prior to tail continuation."""
    dst = DynamicCache(config=text_config)
    for src_layer, dst_layer in zip(src.layers, dst.layers):
        for k, v in vars(src_layer).items():
            setattr(dst_layer, k, v.clone() if isinstance(v, torch.Tensor) else v)
    return dst


class PreallocatedDynamicLayer(DynamicLayer):
    """Fixed-capacity contiguous KV buffer avoiding O(T^2) repeated reallocations during chunked prefill."""

    def __init__(self, max_seq_len: int) -> None:
        super().__init__()
        self.max_seq_len = max_seq_len
        self.cur_len = 0
        self.k_buf = None
        self.v_buf = None

    def lazy_initialization(self, key_states: torch.Tensor, value_states: torch.Tensor) -> None:
        self.dtype, self.device = key_states.dtype, key_states.device
        b, h, _, d = key_states.shape
        self.k_buf = torch.empty((b, h, self.max_seq_len, d), dtype=self.dtype, device=self.device)
        self.v_buf = torch.empty((b, h, self.max_seq_len, d), dtype=self.dtype, device=self.device)
        self.keys = self.k_buf[:, :, :0, :]
        self.values = self.v_buf[:, :, :0, :]
        self.cur_len = 0
        self.is_initialized = True

    def update(self, key_states: torch.Tensor, value_states: torch.Tensor, *args, **kwargs):
        if not self.is_initialized:
            self.lazy_initialization(key_states, value_states)
        n = key_states.shape[-2]
        end = self.cur_len + n
        self.k_buf[:, :, self.cur_len : end, :].copy_(key_states)
        self.v_buf[:, :, self.cur_len : end, :].copy_(value_states)
        self.cur_len = end
        self.keys = self.k_buf[:, :, :end, :]
        self.values = self.v_buf[:, :, :end, :]
        return self.keys, self.values


def make_preallocated_cache(text_config, total_len: int) -> DynamicCache:
    """Allocate a DynamicCache using preallocated layers for full-attention layers."""
    cache = DynamicCache(config=text_config)
    for idx, layer in enumerate(cache.layers):
        if type(layer) is DynamicLayer:
            cache.layers[idx] = PreallocatedDynamicLayer(total_len)
    return cache


def install_flash_chunked_sdpa() -> None:
    """Install causal-lower-right SDPA handling for cached chunked prefill."""
    orig_mask_fn = ALL_MASK_ATTENTION_FUNCTIONS["sdpa"]
    orig_attn_fn = ALL_ATTENTION_FUNCTIONS["sdpa"]

    def fast_sdpa_mask(
        batch_size: int,
        q_length: int,
        kv_length: int,
        q_offset: int = 0,
        kv_offset: int = 0,
        attention_mask=None,
        local_size=None,
        allow_is_causal_skip: bool = True,
        **kwargs,
    ):
        if (
            allow_is_causal_skip
            and attention_mask is None
            and local_size is None
            and kv_offset == 0
            and q_offset + q_length == kv_length
        ):
            if q_length == kv_length or q_length == 1:
                return None
            if 1 < q_length < kv_length:
                return causal_lower_right(q_length, kv_length)
        return orig_mask_fn(
            batch_size=batch_size,
            q_length=q_length,
            kv_length=kv_length,
            q_offset=q_offset,
            kv_offset=kv_offset,
            attention_mask=attention_mask,
            local_size=local_size,
            allow_is_causal_skip=allow_is_causal_skip,
            **kwargs,
        )

    def fast_sdpa_forward(
        module,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attention_mask,
        dropout: float = 0.0,
        scaling=None,
        is_causal=None,
        **kwargs,
    ):
        q_len = query.shape[2]
        kv_len = key.shape[2]
        if (attention_mask is None and 1 < q_len < kv_len) or isinstance(attention_mask, CausalBias):
            bias = attention_mask if isinstance(attention_mask, CausalBias) else causal_lower_right(q_len, kv_len)
            out = torch.nn.functional.scaled_dot_product_attention(
                query,
                key,
                value,
                attn_mask=bias,
                dropout_p=dropout,
                scale=scaling,
                is_causal=False,
                enable_gqa=hasattr(module, "num_key_value_groups"),
            )
            return out.transpose(1, 2).contiguous(), None
        return orig_attn_fn(
            module,
            query,
            key,
            value,
            attention_mask,
            dropout=dropout,
            scaling=scaling,
            is_causal=is_causal,
            **kwargs,
        )

    ALL_MASK_ATTENTION_FUNCTIONS["sdpa"] = fast_sdpa_mask
    ALL_ATTENTION_FUNCTIONS["sdpa"] = fast_sdpa_forward


def chunk_spans(total_len: int, chunk_size: int = CHUNK_SIZE) -> list[tuple[int, int]]:
    """Split a token sequence into `<= chunk_size` chunks, avoiding tiny trailing chunks."""
    spans: list[tuple[int, int]] = []
    pos = 0
    while pos < total_len:
        rem = total_len - pos
        if rem <= chunk_size:
            spans.append((pos, total_len))
            break
        step = chunk_size if (rem - chunk_size) >= 128 else (rem - 128)
        spans.append((pos, pos + step))
        pos += step
    return spans


def install_chunked_cache_pins() -> set[tuple[int, tuple]]:
    """Pin Triton blockdim64 delta-rule keys used when `use_cache=True`."""
    mod = importlib.import_module("fla.ops.common.chunk_delta_h")
    obj = getattr(mod, "chunk_gated_delta_rule_fwd_kernel_h_blockdim64")
    while not isinstance(obj, Autotuner):
        obj = obj.fn
    base_key = (
        32,
        32,
        128,
        128,
        64,
        False,
        "torch.bfloat16",
        "torch.bfloat16",
        "torch.bfloat16",
        "torch.bfloat16",
        "torch.float32",
        "torch.bfloat16",
    )
    dtypes = ("torch.float32", "torch.bfloat16")
    pinned_keys: set[tuple[int, tuple]] = set()
    for d1 in dtypes:
        k1 = base_key + (d1,)
        obj.cache[k1] = triton.Config(
            {"BV": 32}, num_warps=2, num_stages=1, num_ctas=1, maxnreg=None, pre_hook=None, ir_override=None
        )
        pinned_keys.add((id(obj), k1))
        for d2 in dtypes:
            k2 = base_key + (d1, d2)
            obj.cache[k2] = triton.Config(
                {"BV": 32}, num_warps=2, num_stages=1, num_ctas=1, maxnreg=None, pre_hook=None, ir_override=None
            )
            pinned_keys.add((id(obj), k2))
    return pinned_keys


class StaticModelPacker:
    """Tokenizer + W0 readout wrapper for a static pre-merged Transformers checkpoint."""

    def __init__(self, model_dir: Path, readout_path: Path | None) -> None:
        self.tokenizer = AutoTokenizer.from_pretrained(str(model_dir), local_files_only=True, trust_remote_code=False)
        cfg = AutoConfig.from_pretrained(str(model_dir), local_files_only=True, trust_remote_code=False)
        self.text_config = cfg.get_text_config()
        self.native_context = int(self.text_config.max_position_embeddings)

        self.model = Qwen3_5ForCausalLM.from_pretrained(
            str(model_dir),
            config=self.text_config,
            torch_dtype=torch.bfloat16,
            device_map="cuda",
            local_files_only=True,
            trust_remote_code=False,
        )
        self.model.eval()
        self.model.requires_grad_(False)

        self.label_ids = []
        for ch in string.ascii_uppercase:
            ids = self.tokenizer.encode(ch, add_special_tokens=False, truncation=False)
            assert len(ids) == 1
            self.label_ids.append(ids[0])

        if readout_path is not None and readout_path.is_file():
            ro = load_file(str(readout_path))
            self.W = ro["W"].float().cuda()
        else:
            lm_head_w = self.model.lm_head.weight.detach()
            self.W = lm_head_w[self.label_ids].float().cuda()

    def pack(self, state, question: dict) -> dict:
        prompt, names, groups, temp = qwen_prompt(self, state, question)
        ids = self.tokenizer.encode(prompt, add_special_tokens=False, truncation=False)
        if not ids:
            raise ValueError("Encoded prompt cannot be empty")
        if len(ids) > self.native_context:
            raise OverflowError(
                f"Input length {len(ids)} exceeds native context {self.native_context}; silent truncation is forbidden"
            )
        return {
            "ids": ids,
            "names": names,
            "groups": groups,
            "temperature": temp,
            "input_ids_sha256": hashlib.sha256(("[" + ",".join(map(str, ids)) + "]").encode("ascii")).hexdigest(),
        }


class CronusNativeEngine:
    """Thread-safe static-checkpoint Native engine (zero runtime PEFT)."""

    def __init__(
        self,
        model_dir: Path,
        readout_path: Path | None = None,
        calibration_path: Path | None = None,
        manifest_path: Path | None = None,
        profile_path: Path | None = None,
        active_role_label: str = "static",
    ) -> None:
        t_boot = time.perf_counter()
        torch.set_num_threads(8)
        torch.cuda.set_per_process_memory_fraction(CUDA_MEM_FRACTION)
        torch.manual_seed(7813)
        torch.cuda.manual_seed_all(7813)
        torch.set_float32_matmul_precision("highest")
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False

        layout = resolve_bundle_layout(None)
        m_path = manifest_path or layout.manifest_path
        c_path = calibration_path or layout.calibration_path
        ro_path = readout_path or layout.w0_readout_path

        self.manifest_sha = sha_file(m_path)
        self.spec = json.loads(m_path.read_text())
        self.cal = json.loads(c_path.read_text())
        self.cal_sha = sha_file(c_path)
        self.temp_mult = float(self.cal["temperature_multiplier"])
        self.op_threshold = float(self.cal["operating_threshold"])
        self.active_role_label = active_role_label

        orig_run = Autotuner.run
        self.pin_ev = install(str(profile_path or default_config_path("profile_native.json")))
        guarded_run = Autotuner.run
        extra_pinned_keys = install_chunked_cache_pins()

        def combined_pinned_run(instance, *args, **kwargs):
            if len(instance.configs) > 1:
                nargs = dict(zip(instance.arg_names, args))
                all_args = {**nargs, **kwargs}
                kept = {k: v for k, v in all_args.items() if k in instance.arg_names}
                key = tuple(
                    [kept[k] for k in instance.keys if k in kept]
                    + [str(v.dtype) for v in kept.values() if hasattr(v, "dtype")]
                )
                if (id(instance), key) in extra_pinned_keys:
                    assert key in instance.cache, f"Missing cache key {key}"
                    return orig_run(instance, *args, **kwargs)
            return guarded_run(instance, *args, **kwargs)

        Autotuner.run = combined_pinned_run
        install_flash_chunked_sdpa()

        self.packer = StaticModelPacker(model_dir, ro_path)
        assert self.packer.native_context == 262144 and MAX_CONTEXT_TOKENS <= self.packer.native_context
        self.lock = threading.Lock()
        self.prefix_cache: OrderedDict[str, DynamicCache] = OrderedDict()
        self.started_monotonic = time.monotonic()
        self.total_requests = 0
        self.cache_hits = 0

        self._warmup()
        self.cold_load_seconds = time.perf_counter() - t_boot

    def _warmup(self) -> None:
        dummy_state = {"tool": "echo", "args": {"text": "warmup"}, "trusted": {"scope": "authorized"}}
        warm_q = {"type": "noul", "instructions": "Is production harm expected?", "criteria": {"true": "Harm.", "false": "No harm."}}
        lm = self.packer.model.model.language_model
        with torch.inference_mode():
            x = self.packer.pack(dummy_state, warm_q)
            a = torch.tensor([x["ids"]], device="cuda", dtype=torch.long)
            h = lm(input_ids=a, attention_mask=torch.ones_like(a), use_cache=False, return_dict=True).last_hidden_state[0, -1].float()
            _ = torch.stack([torch.logsumexp(self.packer.W[g] @ h, 0) for g in x["groups"]])
            del a, h
            torch.cuda.synchronize()

    def clear_sessions(self, session_key=None) -> int:
        with self.lock:
            if session_key:
                return 1 if self.prefix_cache.pop(session_key, None) is not None else 0
            n = len(self.prefix_cache)
            self.prefix_cache.clear()
            return n

    def health(self) -> dict:
        return {
            "status": "ok",
            "service": "cronus-guard",
            "engine": "native-static",
            "model_name": self.spec.get("model_name", "Cronus-9B"),
            "active_role": self.active_role_label,
            "manifest_sha256": self.manifest_sha,
            "calibration_sha256": self.cal_sha,
            "calibrated_temperature_multiplier": self.temp_mult,
            "calibrated_operating_threshold": self.op_threshold,
            "max_context_tokens": MAX_CONTEXT_TOKENS,
            "native_context_tokens": self.packer.native_context,
            "chunked_prefill_tokens": CHUNK_SIZE,
            "cached_prefixes": len(self.prefix_cache),
            "total_requests": self.total_requests,
            "cache_hits": self.cache_hits,
            "uptime_seconds": round(time.monotonic() - self.started_monotonic, 2),
            "cold_load_seconds": round(self.cold_load_seconds, 2),
            "cuda_allocated_mib": round(torch.cuda.memory_allocated() / 2**20, 2),
            "cuda_reserved_mib": round(torch.cuda.memory_reserved() / 2**20, 2),
            "gpu": torch.cuda.get_device_name(0),
            "node": platform.node(),
            "torch": torch.__version__,
            "transformers": transformers.__version__,
            "no_generation": True,
            "no_silent_truncation": True,
        }

    def evaluate(self, payload: dict) -> dict:
        t_req = time.perf_counter()
        if not isinstance(payload, dict) or "state" not in payload or "question" not in payload:
            raise ValueError("Request JSON must include 'state' and 'question'")

        req_id = str(payload.get("id", "req"))
        state = payload["state"]
        question = payload["question"]
        use_cache = bool(payload.get("use_prefix_cache", False))

        with self.lock, torch.inference_mode():
            x = self.packer.pack(state, question)
            ids = x["ids"]
            if not ids or len(ids) > MAX_CONTEXT_TOKENS:
                raise OverflowError(
                    f"Input length {len(ids)} exceeds max supported context {MAX_CONTEXT_TOKENS}; silent truncation is forbidden"
                )

            head = head_for(question)
            lm = self.packer.model.model.language_model
            cache_hit = False
            cache_key = None
            t_fwd = time.perf_counter()

            prompt_text = None
            if use_cache and len(ids) <= 8192:
                prompt_text, _, _, _ = qwen_prompt(self.packer, state, question)

            if use_cache and prompt_text is not None and "\nQUESTION:\n" in prompt_text:
                partial_ids = self.packer.tokenizer.encode(
                    prompt_text.rsplit("\nQUESTION:\n", 1)[0],
                    add_special_tokens=False,
                    truncation=False,
                )
                n_pref = 0
                for tok_a, tok_b in zip(partial_ids, ids):
                    if tok_a != tok_b:
                        break
                    n_pref += 1
                if 0 < n_pref < len(ids) and n_pref <= CHUNK_SIZE and (len(ids) - n_pref) <= CHUNK_SIZE:
                    prefix_ids = ids[:n_pref]
                    tail_ids = ids[n_pref:]
                    cache_key = digest_obj({"manifest": self.manifest_sha, "role": self.active_role_label, "prefix_ids": prefix_ids})
                    if cache_key in self.prefix_cache:
                        cache_hit = True
                        self.cache_hits += 1
                        self.prefix_cache.move_to_end(cache_key)
                    else:
                        if len(self.prefix_cache) >= MAX_PREFIX_CACHE_ENTRIES:
                            self.prefix_cache.popitem(last=False)
                        pa = torch.tensor([prefix_ids], device="cuda", dtype=torch.long)
                        p_out = lm(input_ids=pa, attention_mask=None, use_cache=True, return_dict=True)
                        self.prefix_cache[cache_key] = p_out.past_key_values
                        del pa

                    past = clone_dynamic_cache(self.prefix_cache[cache_key], lm.config)
                    sfx = torch.tensor([tail_ids], device="cuda", dtype=torch.long)
                    out = lm(input_ids=sfx, attention_mask=None, past_key_values=past, use_cache=True, return_dict=True)
                    h = out.last_hidden_state[0, -1].float()
                    del sfx, past, out
                else:
                    use_cache = False
            else:
                use_cache = False

            if not use_cache:
                if len(ids) <= CHUNK_SIZE:
                    a = torch.tensor([ids], device="cuda", dtype=torch.long)
                    h = lm(input_ids=a, attention_mask=torch.ones_like(a), use_cache=False, return_dict=True).last_hidden_state[0, -1].float()
                    del a
                else:
                    past = make_preallocated_cache(lm.config, len(ids))
                    h = None
                    for c_start, c_end in chunk_spans(len(ids), CHUNK_SIZE):
                        c_ids = torch.tensor([ids[c_start:c_end]], device="cuda", dtype=torch.long)
                        c_out = lm(input_ids=c_ids, attention_mask=None, past_key_values=past, use_cache=True, return_dict=True)
                        if c_end == len(ids):
                            h = c_out.last_hidden_state[0, -1].float()
                        del c_ids, c_out
                        torch.cuda.synchronize()
                    del past

            z = torch.stack([torch.logsumexp(self.packer.W[g] @ h, 0) for g in x["groups"]]) / x["temperature"]
            prob = z.softmax(-1)
            torch.cuda.synchronize()
            fwd_ms = (time.perf_counter() - t_fwd) * 1000.0

            assert torch.isfinite(z).all() and torch.isfinite(prob).all()
            zs = z.cpu().tolist()
            ps = prob.cpu().tolist()
            del h, z, prob

            self.total_requests += 1
            names = x["names"]
            best_idx = max(range(len(ps)), key=ps.__getitem__)
            argmax_label = names[best_idx]

            cal_ps = calibrated_probs(zs, self.temp_mult) if head == "decision" else ps
            gate_action = argmax_label
            native_tau090_action = argmax_label
            proceed_opt = payload.get("proceed_option")
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

            total_ms = (time.perf_counter() - t_req) * 1000.0
            return {
                "id": req_id,
                "status": "ok",
                "task_head": head,
                "active_role": self.active_role_label,
                "names": names,
                "logits": zs,
                "probabilities": ps,
                "calibrated_probabilities": cal_ps,
                "argmax_label": argmax_label,
                "harness_gate_action": gate_action,
                "native_tau090_action": native_tau090_action,
                "operating_temperature": self.temp_mult,
                "operating_threshold": self.op_threshold,
                "input_tokens": len(ids),
                "input_ids_sha256": x["input_ids_sha256"],
                "prefix_cache_used": use_cache,
                "prefix_cache_hit": cache_hit,
                "prefix_cache_key": cache_key,
                "forward_ms": round(fwd_ms, 3),
                "server_ms": round(total_ms, 3),
            }


ENGINE: CronusNativeEngine | None = None


class GuardHTTPHandler(BaseHTTPRequestHandler):
    """HTTP/1.1 JSON handler for Native static server."""

    protocol_version = "HTTP/1.1"

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
    parser.add_argument("--model-dir", type=Path, required=True, help="Static pre-merged HF checkpoint directory (no runtime PEFT).")
    parser.add_argument("--readout-path", type=Path, default=None, help="Path to W0 readout.safetensors.")
    parser.add_argument("--calibration-path", type=Path, default=None)
    parser.add_argument("--manifest-path", type=Path, default=None)
    parser.add_argument("--profile-path", type=Path, default=None)
    parser.add_argument("--role-label", default="static", help="Role label reported in health/evaluate metadata.")
    parser.add_argument("--host", default=os.environ.get("CRONUS_HOST", os.environ.get("GUARD_HOST", "0.0.0.0")))
    parser.add_argument("--port", type=int, default=int(os.environ.get("CRONUS_PORT", os.environ.get("GUARD_PORT", "18971"))))
    return parser.parse_args()


def main() -> None:
    global ENGINE
    args = parse_args()
    ENGINE = CronusNativeEngine(
        model_dir=args.model_dir,
        readout_path=args.readout_path,
        calibration_path=args.calibration_path,
        manifest_path=args.manifest_path,
        profile_path=args.profile_path,
        active_role_label=args.role_label,
    )
    server = ThreadingHTTPServer((args.host, args.port), GuardHTTPHandler)
    print(
        json.dumps(
            {
                "event": "cronus_native_api_ready",
                "host": args.host,
                "port": args.port,
                "max_context_tokens": MAX_CONTEXT_TOKENS,
                "manifest_sha256": ENGINE.manifest_sha,
                "calibration_sha256": ENGINE.cal_sha,
                "temperature_multiplier": ENGINE.temp_mult,
                "operating_threshold": ENGINE.op_threshold,
                "cold_load_seconds": round(ENGINE.cold_load_seconds, 2),
            }
        ),
        flush=True,
    )
    server.serve_forever()


if __name__ == "__main__":
    main()

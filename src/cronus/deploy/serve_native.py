"""Fallback Native Transformers + FLA HTTP API Server for Static Pre-Merged Cronus-9B.

Serves a static pre-merged HF checkpoint (zero runtime PEFT adapters) by
delegating chunked prefill (`CHUNK_SIZE = 2048`), pinned `sm_121` Triton keys,
and `StaticModelRunner` execution to `cronus.static_engine`.
"""

from __future__ import annotations

import argparse
import hashlib
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import importlib.metadata
import json
import math
import os
from pathlib import Path
import platform
import threading
import time
import traceback

import torch

from cronus.paths import resolve_bundle_layout
from cronus.static_engine import CHUNK_SIZE, MAX_CONTEXT_TOKENS, StaticModelRunner
from cronus.task_heads import head_for

CUDA_MEM_FRACTION = float(os.environ.get("CUDA_MEM_FRACTION", "0.33"))
assert 0.0 < CUDA_MEM_FRACTION <= 0.35, CUDA_MEM_FRACTION

MAX_BODY_BYTES = int(os.environ.get("MAX_BODY_BYTES", str(32 * 1024 * 1024)))
MAX_PREFIX_CACHE_ENTRIES = int(os.environ.get("MAX_PREFIX_CACHE_ENTRIES", "8"))


def sha_file(path: Path) -> str:
    """Return SHA-256 hex digest of a file."""
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def calibrated_probs(logits_list: list[float], temp_mult: float) -> list[float]:
    """Compute temperature-scaled softmax probabilities."""
    scaled = [float(v) / temp_mult for v in logits_list]
    mx = max(scaled)
    exps = [math.exp(v - mx) for v in scaled]
    den = math.fsum(exps)
    return [v / den for v in exps]


class CronusNativeEngine:
    """Thread-safe static-checkpoint Native engine delegating forward passes to StaticModelRunner."""

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
        torch.cuda.set_per_process_memory_fraction(CUDA_MEM_FRACTION)

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

        self.runner = StaticModelRunner(
            model_dir=model_dir,
            readout_path=ro_path,
            profile_path=profile_path,
            max_prefix_cache_entries=MAX_PREFIX_CACHE_ENTRIES,
        )
        assert self.runner.native_context == 262144 and MAX_CONTEXT_TOKENS <= self.runner.native_context

        self.lock = threading.Lock()
        self.started_monotonic = time.monotonic()
        self.total_requests = 0
        self.cache_hits = 0

        self._warmup()
        self.cold_load_seconds = time.perf_counter() - t_boot

    def _warmup(self) -> None:
        dummy_state = {"tool": "echo", "args": {"text": "warmup"}, "trusted": {"scope": "authorized"}}
        warm_q = {"type": "noul", "instructions": "Is production harm expected?", "criteria": {"true": "Harm.", "false": "No harm."}}
        packed = self.runner.pack(dummy_state, warm_q)
        _ = self.runner.forward_packed(dummy_state, warm_q, packed, use_prefix_cache=False)

    def clear_sessions(self, session_key=None) -> int:
        with self.lock:
            if session_key:
                return 1 if self.runner.prefix_cache.pop(session_key, None) is not None else 0
            n = len(self.runner.prefix_cache)
            self.runner.prefix_cache.clear()
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
            "native_context_tokens": self.runner.native_context,
            "chunked_prefill_tokens": CHUNK_SIZE,
            "cached_prefixes": len(self.runner.prefix_cache),
            "total_requests": self.total_requests,
            "cache_hits": self.cache_hits,
            "uptime_seconds": round(time.monotonic() - self.started_monotonic, 2),
            "cold_load_seconds": round(self.cold_load_seconds, 2),
            "cuda_allocated_mib": round(torch.cuda.memory_allocated() / 2**20, 2),
            "cuda_reserved_mib": round(torch.cuda.memory_reserved() / 2**20, 2),
            "gpu": torch.cuda.get_device_name(0),
            "node": platform.node(),
            "torch": torch.__version__,
            "transformers": importlib.metadata.version("transformers"),
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

        with self.lock:
            packed = self.runner.pack(state, question)
            fwd = self.runner.forward_packed(
                state,
                question,
                packed,
                use_prefix_cache=use_cache,
                manifest_sha=self.manifest_sha,
                role_label=self.active_role_label,
            )
            if fwd["prefix_cache_hit"]:
                self.cache_hits += 1
            self.total_requests += 1

        head = head_for(question)
        names = packed["names"]
        zs = fwd["logits"]
        ps = fwd["probabilities"]
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
            "input_tokens": len(packed["ids"]),
            "input_ids_sha256": packed["input_ids_sha256"],
            "prefix_cache_used": fwd["prefix_cache_used"],
            "prefix_cache_hit": fwd["prefix_cache_hit"],
            "prefix_cache_key": fwd["prefix_cache_key"],
            "forward_ms": round(fwd["forward_ms"], 3),
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

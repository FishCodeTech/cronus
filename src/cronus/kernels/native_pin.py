"""Pin audited FLA autotune keys in-process for inference; fail closed on unlisted keys."""

from __future__ import annotations

import hashlib
import importlib
import importlib.metadata
import json
import os
from pathlib import Path

import torch
import triton
from triton.runtime.autotuner import Autotuner

from cronus.paths import default_config_path

CUDA_MEM_FRACTION = float(os.environ.get("CUDA_MEM_FRACTION", "0.33"))
EXPECTED_PROFILE_SHA256 = "65127c2cca287c66de8a3e8c6b4fb5afddacc93ac7a48ce41daf9f4c992fa316"


def sha(path: str | Path) -> str:
    """Compute SHA-256 hex digest of a file."""
    with open(path, "rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def ensure_cuda_memory_fraction() -> None:
    """Cap per-process CUDA allocation fraction for unified-memory safety (<= 0.35)."""
    if not (0.0 < CUDA_MEM_FRACTION <= 0.35):
        raise RuntimeError(f"CUDA_MEM_FRACTION={CUDA_MEM_FRACTION} exceeds GB10 safety ceiling 0.35")
    if torch.cuda.is_available():
        torch.cuda.set_per_process_memory_fraction(CUDA_MEM_FRACTION)


def config_dict(c: triton.Config) -> dict:
    """Convert a Triton Config object into a comparable dictionary."""
    return {
        "kwargs": dict(c.kwargs),
        "num_warps": c.num_warps,
        "num_ctas": c.num_ctas,
        "num_stages": c.num_stages,
        "maxnreg": c.maxnreg,
        "pre_hook": None if c.pre_hook is None else str(c.pre_hook),
        "ir_override": c.ir_override,
    }


def _resolve_autotuner(mod, name: str) -> Autotuner:
    obj = getattr(mod, name)
    depth = 0
    while not isinstance(obj, Autotuner) and depth < 8:
        obj = getattr(obj, "fn", None)
        depth += 1
    if not isinstance(obj, Autotuner):
        raise RuntimeError(f"NOT_COVERED autotuner {mod.__name__}.{name}")
    return obj


def install(path: str | Path | None = None):
    """Pin all 261 audited FLA autotuner keys and fail closed on any unlisted key."""
    ensure_cuda_memory_fraction()
    profile_path = Path(path) if path is not None else default_config_path("profile_native.json")
    if sha(profile_path) != EXPECTED_PROFILE_SHA256:
        raise RuntimeError(f"profile hash mismatch: {profile_path}")

    p = json.loads(profile_path.read_text(encoding="utf-8"))
    assert p["profile"] == "NativeNB128_inference_candidate" and len(p["entries"]) == 261
    assert list(torch.cuda.get_device_capability()) == [12, 1]
    for name, version in p["versions"].items():
        assert importlib.metadata.version(name) == version, (name, version)

    from fla.ops.utils.cache import FLA_CACHE_MODE

    if str(FLA_CACHE_MODE).lower().endswith("always"):
        raise RuntimeError("unsupported FLA always mode")

    objects = {}
    active = {}
    used = set()
    single = set()
    for e in p["entries"]:
        mod = importlib.import_module(e["module"])
        assert sha(mod.__file__) == e["module_sha256"]
        obj = _resolve_autotuner(mod, e["name"])
        c = e["config"]
        assert c.get("pre_hook") is None
        cfg = triton.Config(
            c["kwargs"],
            num_warps=c["num_warps"],
            num_stages=c["num_stages"],
            num_ctas=c["num_ctas"],
            maxnreg=c["maxnreg"],
            pre_hook=None,
            ir_override=c["ir_override"],
        )
        key = tuple(e["key"])
        obj.cache[key] = cfg
        objects[(e["module"], e["name"])] = obj
        active[(id(obj), key)] = {"module": e["module"], "name": e["name"], "key": e["key"], "expected": c}
    assert len(objects) == 7 and len(active) == 261

    orig = Autotuner.run

    def guarded(self, *args, **kwargs):
        if len(self.configs) > 1:
            nargs = dict(zip(self.arg_names, args))
            all_args = {**nargs, **kwargs}
            kept = {k: v for k, v in all_args.items() if k in self.arg_names}
            key = tuple([kept[k] for k in self.keys if k in kept] + [str(v.dtype) for v in kept.values() if hasattr(v, "dtype")])
            identity = (id(self), key)
            if identity not in active:
                raise RuntimeError("NOT_COVERED autotune kernel/key " + str((getattr(self, "kernel_name", None), key)))
            if key not in self.cache or config_dict(self.cache[key]) != active[identity]["expected"]:
                raise RuntimeError("PIN_MISMATCH autotune cache")
            used.add(identity)
        else:
            single.add(str(getattr(self, "kernel_name", getattr(self, "base_fn", "single_config"))))
        return orig(self, *args, **kwargs)

    Autotuner.run = guarded

    def evidence(require_all: bool = False) -> dict:
        if require_all and used != set(active):
            raise RuntimeError("NOT_COVERED expected all active keys used: " + str((len(used), len(active))))
        ret = []
        for identity, e in active.items():
            obj = objects[(e["module"], e["name"])]
            key = identity[1]
            actual = config_dict(obj.cache[key])
            assert actual == e["expected"]
            ret.append(
                {
                    "module": e["module"],
                    "name": e["name"],
                    "key": e["key"],
                    "actual": actual,
                    "used": identity in used,
                }
            )
        return {
            "entries": ret,
            "used_count": len(used),
            "single_config_no_benchmark": sorted(single),
            "fallback_forbidden": True,
            "profile_native_candidate_sha256": sha(profile_path),
        }

    return evidence

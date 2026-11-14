"""Pin audited FLA autotune keys in-process for training with backward allowlist."""

from __future__ import annotations

import contextlib
import importlib
import importlib.metadata
import json
import os
from pathlib import Path

import torch
import triton
from triton.runtime.autotuner import Autotuner

from cronus.kernels.native_pin import EXPECTED_PROFILE_SHA256, _resolve_autotuner, config_dict, ensure_cuda_memory_fraction, sha
from cronus.paths import default_config_path

CUDA_MEM_FRACTION = float(os.environ.get("CUDA_MEM_FRACTION", "0.33"))


def install(path: str | Path | None = None, backward_allowlist_path: str | Path | None = None):
    """Pin all 261 forward keys and guard backward Triton autotuner discovery."""
    if not (0.0 < CUDA_MEM_FRACTION <= 0.35):
        raise RuntimeError(f"CUDA_MEM_FRACTION={CUDA_MEM_FRACTION} exceeds GB10 safety ceiling 0.35")
    if torch.cuda.is_available():
        torch.cuda.set_per_process_memory_fraction(CUDA_MEM_FRACTION)
    profile_path = Path(path) if path is not None else default_config_path("profile_native.json")
    allowlist_path = (
        Path(backward_allowlist_path)
        if backward_allowlist_path is not None
        else default_config_path("backward_allowlist.json")
    )
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

    backward = {}
    discovered = {}
    state = {"enabled": False}
    shared_reverse = {}
    for e in json.loads(allowlist_path.read_text(encoding="utf-8")):
        mod = importlib.import_module(e["module"])
        assert sha(mod.__file__) == e["module_sha256"]
        obj = _resolve_autotuner(mod, e["name"])
        assert id(obj) not in {k[0] for k in active}
        backward[id(obj)] = e

    smod = importlib.import_module("fla.ops.utils.cumsum")
    sh = next(e["module_sha256"] for e in p["entries"] if e["module"] == "fla.ops.utils.cumsum")
    assert sha(smod.__file__) == sh
    sobj = _resolve_autotuner(smod, "chunk_local_cumsum_scalar_kernel")
    shared_reverse[id(sobj)] = {
        "module": "fla.ops.utils.cumsum",
        "name": "chunk_local_cumsum_scalar_kernel",
        "module_sha256": sh,
        "semantic": "backward-only reverse=True scalar cumsum",
    }

    orig = Autotuner.run

    def guarded(self, *args, **kwargs):
        if len(self.configs) > 1:
            nargs = dict(zip(self.arg_names, args))
            all_args = {**nargs, **kwargs}
            kept = {k: v for k, v in all_args.items() if k in self.arg_names}
            key = tuple([kept[k] for k in self.keys if k in kept] + [str(v.dtype) for v in kept.values() if hasattr(v, "dtype")])
            identity = (id(self), key)
            if identity not in active:
                reverse_key = (1, 32, 64, False, True, "torch.float32", "torch.float32")
                if not state["enabled"] or not (id(self) in backward or (id(self) in shared_reverse and key == reverse_key)):
                    raise RuntimeError("NOT_COVERED autotune kernel/key " + str((getattr(self, "kernel_name", None), key)))
                result = orig(self, *args, **kwargs)
                if key not in self.cache:
                    raise RuntimeError("backward selected config absent")
                e = backward.get(id(self), shared_reverse.get(id(self)))
                discovered[str((e["module"], e["name"], key))] = {
                    **e,
                    "key": list(key),
                    "config": config_dict(self.cache[key]),
                }
                return result
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
            "profile_native261_sha256": sha(profile_path),
        }

    @contextlib.contextmanager
    def backward_discovery():
        assert not state["enabled"]
        state["enabled"] = True
        try:
            yield
        finally:
            state["enabled"] = False

    def results() -> list[dict]:
        return list(discovered.values())

    return evidence, backward_discovery, results

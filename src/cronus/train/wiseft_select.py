#!/usr/bin/env python3
"""Exact LoRA rank-concatenation WiSE-FT merger and validation evaluator (``w * ckpt_b + (1-w) * ckpt_a``)."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import tempfile
import traceback

from safetensors.torch import load_file, save_file
import torch

from cronus.kernels.native_pin import install as install_native_pin
from cronus.paths import default_config_path
from cronus.role_model import load_candidate, logits
from cronus.train import score_role_validation

CUDA_MEM_FRACTION = float(os.environ.get("CUDA_MEM_FRACTION", "0.33"))


def sha(path: str | Path) -> str:
    """Compute SHA-256 hex digest of a file."""
    with open(path, "rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def concatenate_lora_wiseft(ckpt_a: Path, ckpt_b: Path, output_dir: Path, weight_w: float) -> tuple[str, str]:
    """Exact rank-concatenation interpolation of two rank-8 adapters into a rank-16 adapter."""
    assert 0.0 < weight_w < 1.0, weight_w
    s1 = load_file(str(ckpt_a / "guard.safetensors"))
    s2 = load_file(str(ckpt_b / "guard.safetensors"))
    assert set(s1) == set(s2) and len(s1) == 496

    sa = math.sqrt(1.0 - weight_w)
    sb = math.sqrt(weight_w)
    merged: dict[str, torch.Tensor] = {}
    sample_pair = None
    for k in sorted(s1):
        t1 = s1[k].float()
        t2 = s2[k].float()
        if k.endswith(".lora_A.weight"):
            assert t1.shape[0] == 8 and t2.shape[0] == 8
            merged[k] = torch.cat([sa * t1, sb * t2], dim=0).contiguous()
            if sample_pair is None:
                sample_pair = k[: -len(".lora_A.weight")]
        elif k.endswith(".lora_B.weight"):
            assert t1.shape[1] == 8 and t2.shape[1] == 8
            merged[k] = torch.cat([sa * t1, sb * t2], dim=1).contiguous()
        else:
            raise ValueError(f"unexpected tensor key: {k}")

    assert sample_pair is not None
    ka, kb = sample_pair + ".lora_A.weight", sample_pair + ".lora_B.weight"
    dw_cat = 2.0 * (merged[kb] @ merged[ka])
    dw_exp = 2.0 * ((1.0 - weight_w) * (s1[kb].float() @ s1[ka].float()) + weight_w * (s2[kb].float() @ s2[ka].float()))
    assert torch.allclose(dw_cat, dw_exp, atol=1e-5, rtol=1e-5)

    cfg = json.loads((ckpt_a / "adapter_config.json").read_text(encoding="utf-8"))
    cfg["r"] = 16
    cfg["lora_alpha"] = 32
    cfg["lora_dropout"] = 0.0

    output_dir.mkdir(parents=True, exist_ok=True)
    save_file(merged, str(output_dir / "guard.safetensors"))
    (output_dir / "adapter_config.json").write_text(json.dumps(cfg, indent=2) + "\n", encoding="utf-8")
    return sha(output_dir / "guard.safetensors"), sha(output_dir / "adapter_config.json")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build exact LoRA-concatenated WiSE-FT decision guard and evaluate")
    parser.add_argument("--ckpt-a", type=Path, required=True, help="First epoch directory (weight 1-w)")
    parser.add_argument("--ckpt-b", type=Path, required=True, help="Second epoch directory (weight w)")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--weight-w", type=float, default=0.40)
    parser.add_argument("--data-dir", type=Path, default=None, help="Optional data directory to evaluate validation split")
    parser.add_argument("--base-model-dir", type=Path, default=None)
    parser.add_argument("--source-default-dir", type=Path, default=None)
    parser.add_argument("--w0-readout", type=Path, default=None)
    parser.add_argument("--base-lock", type=Path, default=default_config_path("base_model.lock.json"))
    parser.add_argument("--profile", type=Path, default=default_config_path("profile_native.json"))
    args = parser.parse_args()

    out_dir = args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    if not (0.0 < CUDA_MEM_FRACTION <= 0.35):
        raise RuntimeError(f"CUDA_MEM_FRACTION={CUDA_MEM_FRACTION} exceeds GB10 safety ceiling 0.35")

    try:
        torch.set_num_threads(8)
        torch.set_float32_matmul_precision("highest")
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False

        gsha, csha = concatenate_lora_wiseft(args.ckpt_a, args.ckpt_b, out_dir, args.weight_w)
        if args.data_dir is None:
            print(json.dumps({"weight_w": args.weight_w, "guard_sha256": gsha, "config_sha256": csha}))
            return

        if torch.cuda.is_available():
            torch.cuda.set_per_process_memory_fraction(CUDA_MEM_FRACTION)

        assert args.base_model_dir is not None and args.source_default_dir is not None and args.w0_readout is not None
        manifest_obj = {
            "base_lock_sha256": sha(args.base_lock),
            "source_default_sha256": sha(args.source_default_dir / "adapter_model.safetensors"),
            "W0_readout_sha256": sha(args.w0_readout),
            "adapters": {"decision": {"path": str(out_dir), "guard_sha256": gsha, "config_sha256": csha}},
        }
        with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as tf:
            json.dump(manifest_obj, tf)
            manifest_tmp = Path(tf.name)

        try:
            profile = install_native_pin(args.profile)
            m, _, proof = load_candidate(
                manifest_tmp,
                base_model_dir=args.base_model_dir,
                base_lock=args.base_lock,
                source_default_dir=args.source_default_dir,
                readout_path=args.w0_readout,
                verify_checksums=False,
            )
        finally:
            manifest_tmp.unlink(missing_ok=True)

        splits = json.loads((args.data_dir / "role_splits_v52.json").read_text(encoding="utf-8"))["decision"]
        val = splits["validation"]
        val_set = set(val)
        pack = {
            x["id"]: x
            for x in map(json.loads, (args.data_dir / "pack_v52.jsonl").open(encoding="utf-8"))
            if x["id"] in val_set
        }
        val_rows = [json.loads(line) for line in (args.data_dir / "decision_val_raw.jsonl").open(encoding="utf-8") if line.strip()]
        assert [r["id"] for r in val_rows] == val

        pred = {}
        with torch.inference_mode():
            for r in val_rows:
                z, x, head, names = logits(m, r["state"], r["question"])
                assert head == "decision" and len(names) == 2
                p = z.softmax(-1)
                torch.cuda.synchronize()
                assert torch.isfinite(z).all() and torch.isfinite(p).all()
                pm = pack[r["id"]]
                assert x["input_ids_sha256"] == pm["input_sha256"] and len(x["ids"]) == pm["tokens"]
                pred[r["id"]] = {"z": z.cpu().tolist(), "p": p.cpu().tolist()}

        val_out = score_role_validation("decision", val, pred, pack, splits, f"wiseft_w{args.weight_w:.2f}")
        val_out.update(
            {
                "weight_w": args.weight_w,
                "r": 16,
                "lora_alpha": 32,
                "guard_sha256": gsha,
                "config_sha256": csha,
                "loaded_tensor_proof": proof,
            }
        )
        (out_dir / "validation.json").write_text(json.dumps(val_out, indent=2) + "\n", encoding="utf-8")
        (out_dir / "validation_predictions.json").write_text(
            json.dumps(pred, separators=(",", ":"), allow_nan=False) + "\n", encoding="utf-8"
        )
        (out_dir / "profile_evidence.json").write_text(json.dumps(profile(), indent=2) + "\n", encoding="utf-8")
        print(json.dumps({k: val_out[k] for k in ("weight_w", "macro_family_nll", "SE", "counts", "source_metrics")}))
    except BaseException as exc:
        (out_dir / "failure.json").write_text(
            json.dumps({"error": str(exc), "traceback": traceback.format_exc()}, indent=2) + "\n", encoding="utf-8"
        )
        raise


if __name__ == "__main__":
    main()

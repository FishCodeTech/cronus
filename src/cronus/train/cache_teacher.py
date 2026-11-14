#!/usr/bin/env python3
"""Cache Stage-1 ``default`` teacher logits and probabilities for Stage-2 role training."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import time
import traceback

from safetensors.torch import load_file
import torch

from cronus.kernels.train_pin import install as install_train_pin
from cronus.paths import default_config_path
from cronus.runtime import Classifier

CUDA_MEM_FRACTION = float(os.environ.get("CUDA_MEM_FRACTION", "0.33"))


def sha(path: str | Path) -> str:
    """Compute SHA-256 hex digest of a file."""
    with open(path, "rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description="Cache Stage-1 default teacher predictions for a Stage-2 role")
    parser.add_argument("--role", required=True, choices=("decision", "scope"))
    parser.add_argument("--data-dir", type=Path, required=True, help="Directory with role_splits_v52.json and pack_v52.jsonl")
    parser.add_argument("--base-model-dir", type=Path, required=True, help="Path to Qwen/Qwen3.5-9B weights")
    parser.add_argument("--source-default-dir", type=Path, required=True, help="Stage-1 default LoRA checkpoint directory")
    parser.add_argument("--w0-readout", type=Path, required=True, help="Path to Stage-0 readout.safetensors")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--base-lock", type=Path, default=default_config_path("base_model.lock.json"))
    parser.add_argument("--profile", type=Path, default=default_config_path("profile_native.json"))
    parser.add_argument("--allowlist", type=Path, default=default_config_path("backward_allowlist.json"))
    args = parser.parse_args()

    role = args.role
    out_dir = args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    if not (0.0 < CUDA_MEM_FRACTION <= 0.35):
        raise RuntimeError(f"CUDA_MEM_FRACTION={CUDA_MEM_FRACTION} exceeds GB10 safety ceiling 0.35")

    try:
        torch.cuda.set_per_process_memory_fraction(CUDA_MEM_FRACTION)
        torch.set_num_threads(8)
        torch.set_float32_matmul_precision("highest")
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.manual_seed(7813)
        torch.cuda.manual_seed_all(7813)

        install_train_pin(args.profile, args.allowlist)
        splits = json.loads((args.data_dir / "role_splits_v52.json").read_text(encoding="utf-8"))[role]
        allids = splits["train"] + splits["validation"]
        pack = {
            x["id"]: x
            for x in map(json.loads, (args.data_dir / "pack_v52.jsonl").open(encoding="utf-8"))
            if x["head"] == role
        }
        assert set(allids) == set(pack)

        w0 = load_file(str(args.w0_readout))
        m = Classifier(args.base_model_dir, args.base_lock, adapter=args.source_default_dir, training=False)
        assert torch.equal(m.W.cpu().view(torch.uint8), w0["W0"].view(torch.uint8))
        m.model.base_model.set_adapter(["default"], inference_mode=True)
        m.model.requires_grad_(False)
        m.model.eval()

        teacher = {}
        t0 = time.monotonic()
        with torch.inference_mode():
            for j, item_id in enumerate(allids):
                x = pack[item_id]
                h = m.hidden_state({"ids": x["ids"]})
                z = m.logits_from_hidden(h, {"groups": x["groups"], "temperature": x["nativeT"]})
                p = z.softmax(-1)
                torch.cuda.synchronize()
                assert torch.isfinite(z).all()
                teacher[item_id] = {
                    "z": z.cpu().tolist(),
                    "p": p.cpu().tolist(),
                    "correct": int(z.argmax()) == x["target_index"],
                }
                if (j + 1) % 150 == 0 or (j + 1) == len(allids):
                    print(
                        json.dumps({"role": role, "cached": j + 1, "total": len(allids), "elapsed": round(time.monotonic() - t0, 2)}),
                        flush=True,
                    )

        out_path = out_dir / f"teacher_{role}.json"
        tmp_path = out_dir / f"teacher_{role}.json.tmp"
        tmp_path.write_text(
            json.dumps(
                {"status": "STAGE1_SOURCE_TEACHER_CACHE", "role": role, "count": len(teacher), "examples": teacher},
                separators=(",", ":"),
            )
            + "\n",
            encoding="utf-8",
        )
        tmp_path.replace(out_path)
        print(json.dumps({"role": role, "done": len(teacher), "sha256": sha(out_path)}), flush=True)
    except BaseException as exc:
        err_path = out_dir / f"cache_teacher_{role}_failure.json"
        err_path.write_text(
            json.dumps({"error": str(exc), "traceback": traceback.format_exc()}, indent=2) + "\n", encoding="utf-8"
        )
        raise


if __name__ == "__main__":
    main()

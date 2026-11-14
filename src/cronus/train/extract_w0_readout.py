#!/usr/bin/env python3
"""Stage-0 feature and frozen 26x4096 FP32 W0 readout extractor."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import struct
import time
import traceback

from safetensors.torch import save_file
import torch

from cronus.kernels.train_pin import install as install_train_pin
from cronus.paths import default_config_path
from cronus.runtime import Classifier

CUDA_MEM_FRACTION = float(os.environ.get("CUDA_MEM_FRACTION", "0.33"))


def sha(path: str | Path) -> str:
    """Compute SHA-256 hex digest of a file."""
    with open(path, "rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def _load_rows(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def _floatbits_equal(a: list[float], b: list[float]) -> bool:
    return len(a) == len(b) and all(struct.pack(">d", float(x)) == struct.pack(">d", float(y)) for x, y in zip(a, b))


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract frozen Stage-0 hidden features and W0 readout")
    parser.add_argument("--input-dir", type=Path, required=True, help="Directory with records.jsonl, order.jsonl, packs, and protocol.json")
    parser.add_argument("--output-dir", type=Path, required=True, help="Destination directory for hidden states and readout.safetensors")
    parser.add_argument("--base-model-dir", type=Path, required=True, help="Path to Qwen/Qwen3.5-9B weights")
    parser.add_argument("--base-lock", type=Path, default=default_config_path("base_model.lock.json"))
    parser.add_argument("--profile", type=Path, default=default_config_path("profile_native.json"))
    parser.add_argument("--allowlist", type=Path, default=default_config_path("backward_allowlist.json"))
    parser.add_argument("--adapter-dir", type=Path, default=None, help="Optional LoRA adapter directory")
    args = parser.parse_args()

    if not (0.0 < CUDA_MEM_FRACTION <= 0.35):
        raise RuntimeError(f"CUDA_MEM_FRACTION={CUDA_MEM_FRACTION} exceeds GB10 safety ceiling 0.35")

    in_dir = args.input_dir
    out_dir = args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    try:
        torch.cuda.set_per_process_memory_fraction(CUDA_MEM_FRACTION)
        torch.set_float32_matmul_precision("highest")
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.set_num_threads(8)
        torch.manual_seed(7813)
        torch.cuda.manual_seed_all(7813)

        proto_path = in_dir / "protocol.json"
        if proto_path.exists():
            proto = json.loads(proto_path.read_text(encoding="utf-8"))
            if proto.get("status") != "FROZEN_AUTHORIZED_FOR_FEATURE_EXTRACTION" or proto.get("authorized_extraction") is not True:
                raise RuntimeError("NOT_READY candidate is not authorized")

        pin, _, _ = install_train_pin(args.profile, args.allowlist)
        m = Classifier(args.base_model_dir, args.base_lock, adapter=args.adapter_dir, training=False)
        m.model.eval()
        assert not m.model.training and all(not x.requires_grad for x in m.model.parameters())
        assert m.W.dtype == torch.float32 and m.W.shape == (26, 4096) and m.bias is None

        raw = _load_rows(in_dir / "records.jsonl")
        pack = _load_rows(in_dir / "train_pack.jsonl")
        order = _load_rows(in_dir / "order.jsonl")
        dev = _load_rows(in_dir / "unlabeled.jsonl")
        dp = _load_rows(in_dir / "dev_pack.jsonl")
        ref_path = in_dir / "reference125.jsonl"
        refs = _load_rows(ref_path) if ref_path.exists() else []

        by = {x["id"]: x for x in raw}
        pi = {x["id"]: x for x in pack}
        di = {x["id"]: x for x in dp}
        ri = {x["id"]: x for x in refs}

        def run_phase(name: str, examples: list[dict], index: dict[str, dict], is_train: bool) -> dict:
            phase_dir = out_dir / name
            phase_dir.mkdir(parents=True, exist_ok=False)
            hidden_dim = m.W.shape[1]
            hs = torch.empty((len(examples), hidden_dim), dtype=torch.float32, device="cpu")
            meta = []
            count = 0
            with torch.inference_mode():
                for e in examples:
                    ref = index[e["id"]]
                    x = m.pack(e["state"], e["question"])
                    key = "input_sha256" if is_train else "inputSHA"
                    assert x["input_ids_sha256"] == ref[key] and len(x["ids"]) == ref["tokens"]
                    assert x["names"] == ref["names"] and x["groups"] == ref["groups"]
                    h = m.hidden_state(x)
                    z = m.logits_from_hidden(h, x)
                    p = z.softmax(-1)
                    torch.cuda.synchronize()
                    assert h.shape == (hidden_dim,) and h.dtype == z.dtype == torch.float32
                    assert torch.isfinite(h).all() and torch.isfinite(z).all() and torch.isfinite(p).all()
                    if not is_train and e["id"] in ri:
                        old = ri[e["id"]]
                        assert _floatbits_equal(z.cpu().tolist(), old["logits"]) and _floatbits_equal(
                            p.cpu().tolist(), old["probabilities"]
                        )
                    hs[count].copy_(h.cpu())
                    meta.append(
                        {
                            "id": e["id"],
                            "inputSHA": x["input_ids_sha256"],
                            "names": x["names"],
                            "groups": x["groups"],
                            "nativeT": x["temperature"],
                            "type": ref["type"],
                            "tokens": len(x["ids"]),
                            "row": count,
                        }
                    )
                    count += 1
                    if count % 100 == 0:
                        print(json.dumps({"event": "progress", "phase": name, "count": count}), flush=True)
            save_file({"hidden": hs}, str(phase_dir / "hidden.safetensors"))
            (phase_dir / "metadata.jsonl").write_text(
                "".join(json.dumps(v, separators=(",", ":")) + "\n" for v in meta), encoding="utf-8"
            )
            return {
                "count": count,
                "hidden_sha256": sha(phase_dir / "hidden.safetensors"),
                "metadata_sha256": sha(phase_dir / "metadata.jsonl"),
            }

        t0 = time.monotonic()
        dev_res = run_phase("development", dev, di, False)
        train_res = run_phase("training", [by[x["id"]] for x in order], pi, True)
        save_file(
            {"W0": m.W.detach().cpu().contiguous(), "label_ids": torch.tensor(m.label_ids, dtype=torch.int64)},
            str(out_dir / "readout.safetensors"),
        )
        evidence = pin()
        (out_dir / "forward_config_evidence.json").write_text(json.dumps(evidence, indent=2) + "\n", encoding="utf-8")
        receipt = {
            "status": "STAGE0_FEATURE_READOUT_COMPLETE",
            "dev": dev_res,
            "train": train_res,
            "readout_sha256": sha(out_dir / "readout.safetensors"),
            "forward_config_evidence_sha256": sha(out_dir / "forward_config_evidence.json"),
            "elapsed_seconds": time.monotonic() - t0,
        }
        (out_dir / "receipt.json").write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
        print(json.dumps({"event": "complete", "dev": dev_res["count"], "train": train_res["count"]}), flush=True)
    except BaseException as exc:
        (out_dir / "failure.json").write_text(
            json.dumps({"status": "FAILED", "error_type": type(exc).__name__, "message": str(exc), "traceback": traceback.format_exc()}, indent=2)
            + "\n",
            encoding="utf-8",
        )
        raise


if __name__ == "__main__":
    main()

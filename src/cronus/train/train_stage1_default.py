#!/usr/bin/env python3
"""Stage-1 ``default`` adapter fine-tuning (`CE + 0.1*Brier + 0.1*KL` retention)."""

from __future__ import annotations

import argparse
import collections
import hashlib
import json
import os
from pathlib import Path
import random
import time

from safetensors import safe_open
from safetensors.torch import load_file
import torch
import torch.nn.functional as F

from cronus.kernels.train_pin import install as install_train_pin
from cronus.paths import default_config_path
from cronus.runtime import Classifier

CUDA_MEM_FRACTION = float(os.environ.get("CUDA_MEM_FRACTION", "0.33"))


def sha(path: str | Path) -> str:
    """Compute SHA-256 hex digest of a file."""
    with open(path, "rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def _rows(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def _atomic_json(path: Path, value) -> None:
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train Stage-1 default LoRA adapter on Qwen3.5-9B")
    parser.add_argument("--data-dir", type=Path, required=True, help="Directory with records.jsonl, order.jsonl, masks.jsonl, cpu_pack/pack.jsonl")
    parser.add_argument("--model-dir", type=Path, required=True, help="Path to Qwen/Qwen3.5-9B weights")
    parser.add_argument("--cache-dir", type=Path, required=True, help="Stage-0 hidden state and readout cache directory")
    parser.add_argument("--output-dir", type=Path, required=True, help="Destination directory for default checkpoints")
    parser.add_argument("--initial-adapter", type=Path, default=None, help="Optional warmup LoRA checkpoint directory")
    parser.add_argument("--fast-dir", type=Path, default=None, help="Optional fast validation payload directory")
    parser.add_argument("--base-lock", type=Path, default=default_config_path("base_model.lock.json"))
    parser.add_argument("--profile", type=Path, default=default_config_path("profile_native.json"))
    parser.add_argument("--allowlist", type=Path, default=default_config_path("backward_allowlist.json"))
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--warmup-steps", type=int, default=5)
    parser.add_argument("--max-steps", type=int, default=0, help="0 uses one full pass over order.jsonl")
    parser.add_argument("--save-every", type=int, default=100)
    args = parser.parse_args()

    if not (0.0 < CUDA_MEM_FRACTION <= 0.35):
        raise RuntimeError(f"CUDA_MEM_FRACTION={CUDA_MEM_FRACTION} exceeds GB10 safety ceiling 0.35")
    torch.cuda.set_per_process_memory_fraction(CUDA_MEM_FRACTION)
    torch.set_float32_matmul_precision("highest")
    random.seed(7813)
    torch.manual_seed(7813)
    torch.cuda.manual_seed_all(7813)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_num_threads(8)

    data_dir = args.data_dir
    out_dir = args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    raw = _rows(data_dir / "records.jsonl")
    packs = _rows(data_dir / "cpu_pack" / "pack.jsonl")
    masks = _rows(data_dir / "masks.jsonl")
    order = _rows(data_dir / "order.jsonl")

    by = {x["id"]: x for x in raw}
    pi = {x["id"]: x for x in packs}
    masked = {x["id"] for x in masks}
    assert {x["id"] for x in order} == set(by) - masked

    replay_ids = {x["id"] for x in order if x.get("lane") == "RETENTION"}
    general_ids = {x["id"] for x in order if by[x["id"]].get("pool") == "general"}
    teacher_ids = replay_ids | general_ids

    pin_evidence, backward_context, backward_results = install_train_pin(args.profile, args.allowlist)
    m = Classifier(args.model_dir, args.base_lock, adapter=args.initial_adapter, training=True)
    params = [p for n, p in m.model.named_parameters() if p.requires_grad]
    assert len(params) == 176 and all("lora_" in n for n, _ in m.model.named_parameters() if p.requires_grad)

    def save_ckpt(step_idx: int) -> dict:
        ckpt_dir = out_dir / f"checkpoint-{step_idx}"
        m.save_adapter(ckpt_dir)
        with safe_open(str(ckpt_dir / "adapter_model.safetensors"), framework="pt", device="cpu") as f:
            keys = list(f.keys())
            assert len(keys) == 176 and all("lora_" in n and torch.isfinite(f.get_tensor(n)).all() for n in keys)
        rec = {
            "status": "checkpoint_complete",
            "step": step_idx,
            "adapter_sha256": sha(ckpt_dir / "adapter_model.safetensors"),
            "adapter_config_sha256": sha(ckpt_dir / "adapter_config.json"),
            "all_finite": True,
            "tensor_count": len(keys),
        }
        _atomic_json(ckpt_dir / "receipt.json", rec)
        return rec

    save_ckpt(0)

    cache_readout = load_file(str(args.cache_dir / "readout.safetensors"))
    h_all = load_file(str(args.cache_dir / "training" / "hidden.safetensors"))["hidden"]
    meta = _rows(args.cache_dir / "training" / "metadata.jsonl")
    mi = {x["id"]: x for x in meta}
    assert torch.equal(cache_readout["W0"], m.W.cpu()) and cache_readout["label_ids"].tolist() == m.label_ids

    teacher = {}
    source_correct = set()
    m.model.eval()
    with torch.inference_mode():
        for item in order:
            item_id = item["id"]
            if item_id in teacher_ids:
                z_meta = mi[item_id]
                v = h_all[z_meta["row"]].to("cuda")
                logits_v = torch.stack([torch.logsumexp(m.W[g] @ v, dim=0) for g in z_meta["groups"]]) / z_meta["nativeT"]
                prob = logits_v.softmax(-1)
                gold = pi[item_id]["target_index"]
                pred = int(logits_v.argmax().item())
                if pred == gold:
                    source_correct.add(item_id)
                teacher[item_id] = {
                    "names": z_meta["names"],
                    "input_sha256": z_meta["inputSHA"],
                    "probabilities": prob.cpu().tolist(),
                }
    m.model.train()

    fast_rows = _rows(args.fast_dir / "payload.jsonl") if args.fast_dir is not None else []
    fast_pack = {x["id"]: x for x in _rows(args.fast_dir / "packing.jsonl")} if args.fast_dir is not None else {}

    def eval_fast(step_idx: int, ck: dict) -> None:
        if not fast_rows:
            return
        dest = out_dir / f"fast-{step_idx}"
        dest.mkdir(parents=True, exist_ok=True)
        mode_prev = m.model.training
        m.model.eval()
        with (dest / "predictions.jsonl").open("w", encoding="utf-8") as f, torch.inference_mode():
            for e in fast_rows:
                ref = fast_pack[e["id"]]
                x = m.pack(e["state"], e["question"])
                assert x["input_ids_sha256"] == ref["inputSHA"]
                z = m.logits(x)
                p = z.softmax(-1)
                torch.cuda.synchronize()
                f.write(
                    json.dumps(
                        {
                            "id": e["id"],
                            "step": step_idx,
                            "adapter_sha256": ck["adapter_sha256"],
                            "names": x["names"],
                            "logits": z.cpu().tolist(),
                            "probabilities": p.cpu().tolist(),
                        },
                        separators=(",", ":"),
                    )
                    + "\n"
                )
        m.model.train(mode_prev)

    total_steps = args.max_steps if args.max_steps > 0 else (len(order) + args.batch_size - 1) // args.batch_size
    opt = torch.optim.AdamW(params, lr=args.lr, betas=(0.9, 0.95), eps=1e-8, weight_decay=args.weight_decay)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt,
        lambda i: min(1.0, (i + 1) / args.warmup_steps)
        if i < args.warmup_steps
        else max(0.0, (total_steps - i) / max(1, total_steps - args.warmup_steps)),
    )

    seen = []
    started = time.monotonic()
    totals = collections.defaultdict(lambda: collections.Counter())
    for step_idx in range(1, total_steps + 1):
        opt.zero_grad(set_to_none=True)
        batch = order[(step_idx - 1) * args.batch_size : step_idx * args.batch_size]
        for item in batch:
            e = by[item["id"]]
            ref = pi[e["id"]]
            x = m.pack(e["state"], e["question"])
            assert x["input_ids_sha256"] == ref["input_sha256"]
            hard_target = torch.tensor(ref["target_distribution"], device="cuda", dtype=torch.float32)
            z = m.logits(x)
            lp = z.log_softmax(-1)
            p = lp.exp()
            kl = torch.zeros((), device=z.device, dtype=z.dtype)
            if e["id"] in source_correct:
                t = teacher[e["id"]]
                target = torch.tensor(t["probabilities"], device="cuda", dtype=torch.float32)
                kl = F.kl_div(lp, target, reduction="sum")
            else:
                target = hard_target
            ce = -(target * lp).sum()
            brier = ((p - target) ** 2).sum()
            loss = ce + 0.1 * brier + 0.1 * kl
            if not torch.isfinite(loss):
                raise RuntimeError("nonfinite loss")
            with backward_context():
                (loss / len(batch)).backward()
            totals["all"]["count"] += 1
            totals["all"]["loss_sum"] += float(loss.detach())
            seen.append(e["id"])

        assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in params)
        grad_norm = torch.nn.utils.clip_grad_norm_(params, 1.0)
        assert torch.isfinite(grad_norm)
        opt.step()
        sched.step()
        if step_idx == 1 or step_idx % 10 == 0:
            print(
                json.dumps(
                    {
                        "event": "progress",
                        "step": step_idx,
                        "gradient_norm": float(grad_norm),
                        "elapsed_seconds": round(time.monotonic() - started, 2),
                    }
                ),
                flush=True,
            )
        if step_idx == 1 or step_idx % args.save_every == 0 or step_idx == total_steps:
            ck = save_ckpt(step_idx)
            if step_idx % args.save_every == 0 or step_idx == total_steps:
                eval_fast(step_idx, ck)

    _atomic_json(out_dir / "backend-evidence.json", {"forward": pin_evidence(), "backward_discovered": backward_results()})
    _atomic_json(
        out_dir / "completion.json",
        {
            "status": "stage1_default_complete",
            "steps": total_steps,
            "used": len(seen),
            "mean_loss": totals["all"]["loss_sum"] / max(1, totals["all"]["count"]),
        },
    )


if __name__ == "__main__":
    main()

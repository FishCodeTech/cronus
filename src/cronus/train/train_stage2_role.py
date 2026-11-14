#!/usr/bin/env python3
"""Stage-2 all-linear (``r=8``, 248 modules) role LoRA trainer + 1-SE epoch selector."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import random
import struct
import time
import traceback

from peft import LoraConfig, get_peft_model_state_dict
from peft.tuners.tuners_utils import BaseTunerLayer
from safetensors.torch import load_file, save_file
import torch
import torch.nn.functional as F

from cronus.kernels.train_pin import install as install_train_pin
from cronus.paths import default_config_path
from cronus.runtime import Classifier
from cronus.train import score_role_validation

CUDA_MEM_FRACTION = float(os.environ.get("CUDA_MEM_FRACTION", "0.33"))


def sha(path: str | Path) -> str:
    """Compute SHA-256 hex digest of a file."""
    with open(path, "rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def balanced_order(ids: list[str], pack: dict[str, dict], seed: int) -> list[str]:
    """Interleave semantic classes via smooth deterministic deficit scheduling."""
    rng = random.Random(seed)
    by_cls: dict[str, list[str]] = {}
    for item_id in ids:
        by_cls.setdefault(pack[item_id]["semantic_class"], []).append(item_id)
    for cls_ids in by_cls.values():
        rng.shuffle(cls_ids)
    queues = {c: list(v) for c, v in sorted(by_cls.items())}
    tot = {c: len(v) for c, v in queues.items()}
    total_n = sum(tot.values())
    order: list[str] = []
    taken = {c: 0 for c in queues}
    for k in range(total_n):
        active = [c for c, q in queues.items() if q]
        chosen = max(active, key=lambda c: ((k + 1) * (tot[c] / total_n) - taken[c], -len(queues[c]), c))
        order.append(queues[chosen].pop())
        taken[chosen] += 1
    assert len(order) == len(ids) and set(order) == set(ids)
    return order


def _atomic_write(out_dir: Path, name: str, value) -> None:
    tmp = out_dir / (name + ".tmp")
    tmp.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    tmp.replace(out_dir / name)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train Stage-2 all-linear role adapter (decision or scope)")
    parser.add_argument("--role", required=True, choices=("decision", "scope"))
    parser.add_argument("--data-dir", type=Path, required=True, help="Directory with role_splits_v52.json, pack_v52.jsonl, teacher_<role>.json")
    parser.add_argument("--base-model-dir", type=Path, required=True, help="Path to Qwen/Qwen3.5-9B weights")
    parser.add_argument("--source-default-dir", type=Path, required=True, help="Stage-1 default LoRA directory")
    parser.add_argument("--w0-readout", type=Path, required=True, help="Path to Stage-0 readout.safetensors")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--teacher-path", type=Path, default=None, help="Override path to cached teacher_<role>.json")
    parser.add_argument("--base-lock", type=Path, default=default_config_path("base_model.lock.json"))
    parser.add_argument("--profile", type=Path, default=default_config_path("profile_native.json"))
    parser.add_argument("--allowlist", type=Path, default=default_config_path("backward_allowlist.json"))
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=2.5e-5)
    args = parser.parse_args()

    role = args.role
    out_dir = args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    epoch = 0
    step = 0
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
        random.seed(7813)

        pin, bwd, back_results = install_train_pin(args.profile, args.allowlist)
        splits = json.loads((args.data_dir / "role_splits_v52.json").read_text(encoding="utf-8"))[role]
        train = splits["train"]
        val = splits["validation"]
        allids = train + val
        assert len(allids) == len(set(allids)) == splits["all_n"] and not (set(train) & set(val))

        pack = {
            x["id"]: x
            for x in map(json.loads, (args.data_dir / "pack_v52.jsonl").open(encoding="utf-8"))
            if x["head"] == role
        }
        teacher_file = args.teacher_path or (args.data_dir / f"teacher_{role}.json")
        teacher = json.loads(teacher_file.read_text(encoding="utf-8"))["examples"]
        assert set(allids) <= set(pack) and set(allids) <= set(teacher)

        w0 = load_file(str(args.w0_readout))
        m = Classifier(args.base_model_dir, args.base_lock, adapter=args.source_default_dir, training=False)
        assert torch.equal(m.W.cpu().view(torch.uint8), w0["W0"].view(torch.uint8))
        assert m.label_ids == w0["label_ids"].tolist() and m.bias is None
        original = load_file(str(args.source_default_dir / "adapter_model.safetensors"))

        frozen_refs = list(m.model.named_parameters()) + list(m.model.named_buffers())

        def logical_hash() -> str:
            h = hashlib.sha256()
            for name, p in frozen_refs:
                h.update(name.encode())
                h.update(str(p.dtype).encode())
                h.update(str(tuple(p.shape)).encode())
                h.update(p.detach().cpu().contiguous().reshape(-1).view(torch.uint8).numpy().tobytes())
            return h.hexdigest()

        before = logical_hash()

        def verify_default() -> None:
            actual = get_peft_model_state_dict(m.model, adapter_name="default")
            assert set(actual) == set(original) and len(actual) == 176
            for n in actual:
                assert torch.equal(actual[n].detach().cpu().contiguous().view(torch.uint8), original[n].contiguous().view(torch.uint8)), n

        verify_default()

        cfg = LoraConfig(r=8, lora_alpha=16, lora_dropout=0.05, target_modules="all-linear", bias="none", task_type="CAUSAL_LM")
        m.model.add_adapter("guard", cfg, autocast_adapter_dtype=True)
        m.model.requires_grad_(False)
        guard_layers = [
            n for n, mod in m.model.named_modules() if isinstance(mod, BaseTunerLayer) and hasattr(mod, "lora_A") and "guard" in mod.lora_A
        ]
        assert len(guard_layers) == 248, len(guard_layers)
        guard = [p for n, p in m.model.named_parameters() if ".guard." in n and "lora_" in n]
        assert len(guard) == 496 and all(p.dtype == torch.float32 for p in guard)
        for n, p in m.model.named_parameters():
            if "lora_B.guard" in n:
                assert not torch.count_nonzero(p)

        def set_mode(use_guard: bool = False, training: bool = False) -> None:
            m.model.base_model.set_adapter(["default", "guard"] if use_guard else ["default"], inference_mode=not training)
            for n, p in m.model.named_parameters():
                p.requires_grad_(training and "lora_" in n and ".guard." in n)
            m.model.train(training)
            if training:
                for mod in m.model.modules():
                    if hasattr(mod, "lora_dropout") and "default" in mod.lora_dropout:
                        mod.lora_dropout["default"].eval()

        def forward_packed(item_id: str) -> torch.Tensor:
            x = pack[item_id]
            h = m.hidden_state({"ids": x["ids"]})
            return m.logits_from_hidden(h, {"groups": x["groups"], "temperature": x["nativeT"]})

        # Verify zero-init guard exactly matches frozen Stage-1 default teacher
        set_mode(True, False)
        anchors = sorted(allids, key=lambda i: hashlib.sha256(("cronus-source0-" + i).encode()).hexdigest())[:11]
        anchors.append(max(allids, key=lambda i: pack[i]["tokens"]))
        anchors = list(dict.fromkeys(anchors))

        def ieee_same(a: list[float], b: list[float]) -> bool:
            return len(a) == len(b) and all(struct.pack(">d", float(u)) == struct.pack(">d", float(v)) for u, v in zip(a, b))

        with torch.inference_mode():
            for aid in anchors:
                probs = forward_packed(aid).softmax(-1).cpu().tolist()
                torch.cuda.synchronize()
                assert ieee_same(probs, teacher[aid]["p"]), aid
        _atomic_write(out_dir, "source0_anchors.json", {"status": "ALL_LINEAR_ZERO_GUARD_IEEE_EXACT", "count": len(anchors), "ids": anchors})

        history = [score_role_validation(role, val, teacher, pack, splits, 0)]
        _atomic_write(out_dir, "validation_epoch0.json", history[0])

        m.model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        m.model.enable_input_require_grads()
        torch.manual_seed(7813)
        torch.cuda.manual_seed_all(7813)
        set_mode(True, True)

        per_epoch = math.ceil(len(train) / args.batch_size)
        maxsteps = args.epochs * per_epoch
        warmup = 15
        opt = torch.optim.AdamW(guard, lr=args.lr, betas=(0.9, 0.95), eps=1e-8, weight_decay=0.02)
        sched = torch.optim.lr_scheduler.LambdaLR(
            opt,
            lambda k: min(1.0, (k + 1) / warmup)
            if k < warmup
            else max(0.05, 0.5 * (1.0 + math.cos(math.pi * (k - warmup) / max(1, maxsteps - warmup)))),
        )

        def save_epoch(ep: int) -> None:
            ckpt = out_dir / f"epoch-{ep}"
            ckpt.mkdir(exist_ok=True)
            state = get_peft_model_state_dict(m.model, adapter_name="guard")
            save_file({k: v.detach().cpu().contiguous() for k, v in state.items()}, str(ckpt / "guard.safetensors"))
            m.model.peft_config["guard"].save_pretrained(str(ckpt))
            rec = {
                "status": "ALL_LINEAR_ROLE_EPOCH_CHECKPOINT",
                "role": role,
                "epoch": ep,
                "global_step": step,
                "guard_tensors": 496,
                "guard_sha256": sha(ckpt / "guard.safetensors"),
                "adapter_config_sha256": sha(ckpt / "adapter_config.json"),
                "source_default_sha256": sha(args.source_default_dir / "adapter_model.safetensors"),
                "W0_sha256": sha(args.w0_readout),
            }
            (ckpt / "receipt.json").write_text(json.dumps(rec, indent=2) + "\n", encoding="utf-8")

        save_epoch(0)
        loss_sum = 0.0
        used: list[str] = []
        start = time.monotonic()

        for epoch in range(1, args.epochs + 1):
            set_mode(True, True)
            order = balanced_order(train, pack, 7813 + epoch)
            epoch_used = []
            for b in range(per_epoch):
                opt.zero_grad(set_to_none=True)
                batch = order[b * args.batch_size : (b + 1) * args.batch_size]
                for item_id in batch:
                    assert item_id not in val
                    x = pack[item_id]
                    y_idx = x["target_index"]
                    z = forward_packed(item_id)
                    y = torch.tensor(x["target_distribution"], device="cuda", dtype=torch.float32)
                    lp = z.log_softmax(-1)
                    p = lp.exp()
                    loss = -(y * lp).sum() + 0.15 * ((p - y) ** 2).sum()
                    if (not x["is_guardtask"]) and teacher[item_id]["correct"] and teacher[item_id]["p"][y_idx] >= 0.95:
                        loss = loss + 0.05 * F.kl_div(lp, torch.tensor(teacher[item_id]["p"], device="cuda", dtype=torch.float32), reduction="sum")
                    if role == "decision" and y_idx != x["proceed_index"]:
                        p_proc = p[x["proceed_index"]]
                        loss = loss + 0.25 * (-torch.log((1.0 - p_proc).clamp_min(1e-6)))
                    elif role == "scope" and y_idx == 1:
                        loss = 1.30 * loss
                    assert torch.isfinite(loss)
                    with bwd():
                        (loss / len(batch)).backward()
                    epoch_used.append(item_id)
                    used.append(item_id)
                    loss_sum += float(loss.detach())

                assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in guard)
                norm = torch.nn.utils.clip_grad_norm_(guard, 1.0)
                assert torch.isfinite(norm)
                opt.step()
                sched.step()
                step += 1
                if step == 1:
                    verify_default()
                    _atomic_write(out_dir, "first_update.json", {"role": role, "gradient_norm": float(norm), "guard_tensors": 496})

            assert len(epoch_used) == len(train) and set(epoch_used) == set(train)
            save_epoch(epoch)
            set_mode(True, False)
            pred = {}
            with torch.inference_mode():
                for val_id in val:
                    z = forward_packed(val_id)
                    p = z.softmax(-1)
                    torch.cuda.synchronize()
                    assert torch.isfinite(z).all()
                    pred[val_id] = {"z": z.cpu().tolist(), "p": p.cpu().tolist()}
            result = score_role_validation(role, val, pred, pack, splits, epoch)
            history.append(result)
            _atomic_write(out_dir, f"validation_epoch{epoch}.json", result)
            _atomic_write(out_dir, f"validation_predictions_epoch{epoch}.json", pred)
            print(json.dumps({k: v for k, v in result.items() if k not in ("family_nll", "frontier_sample")}), flush=True)

        best = min(history, key=lambda x: x["macro_family_nll"])
        cutoff = best["macro_family_nll"] + best["SE"]
        cands = [x for x in history if x["macro_family_nll"] <= cutoff]
        if role == "decision":
            sel_ep = min(
                cands,
                key=lambda x: (
                    x["counts"].get("nonproceed_autoallow", 0),
                    x["counts"].get("proceed_falseblock_rate", 1.0),
                    x["macro_family_nll"],
                    x["epoch"],
                ),
            )["epoch"]
        else:
            sel_ep = min(
                cands,
                key=lambda x: (
                    0 if (x["counts"].get("recall") or 0) >= 0.90 else 1,
                    x["counts"].get("fpr", 1.0),
                    x["macro_family_nll"],
                    x["epoch"],
                ),
            )["epoch"]

        assert logical_hash() == before
        verify_default()
        assert torch.equal(m.W.cpu().view(torch.uint8), w0["W0"].view(torch.uint8))
        _atomic_write(out_dir, "backend_evidence.json", {"forward": pin(), "backward": back_results()})
        _atomic_write(
            out_dir,
            "selection.json",
            {
                "status": "STAGE2_1SE_ROLE_SELECTED",
                "role": role,
                "selected_epoch": sel_ep,
                "best_mean_epoch": best["epoch"],
                "one_SE_cutoff": cutoff,
                "all_metrics": history,
                "candidate_guard_sha256": sha(out_dir / f"epoch-{sel_ep}/guard.safetensors"),
                "candidate_config_sha256": sha(out_dir / f"epoch-{sel_ep}/adapter_config.json"),
            },
        )
    except BaseException as exc:
        _atomic_write(
            out_dir,
            "failure.json",
            {"role": role, "epoch": epoch, "step": step, "error": str(exc), "traceback": traceback.format_exc()},
        )
        raise


if __name__ == "__main__":
    main()

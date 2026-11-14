"""Verify adapter reload IEEE parity and decision option-order invariance on short prompts."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import torch

from cronus.eval import load_jsonl, same_ieee
from cronus.kernels.native_pin import install
from cronus.paths import default_config_path
from cronus.role_model import activate, load_candidate, logits

CUDA_MEM_FRACTION = float(os.environ.get("CUDA_MEM_FRACTION", "0.33"))
assert 0.0 < CUDA_MEM_FRACTION <= 0.35, CUDA_MEM_FRACTION


def run_reload_parity(args: argparse.Namespace) -> None:
    """Compare freshly reloaded adapter probabilities against reference JSON."""
    torch.set_num_threads(8)
    torch.cuda.set_per_process_memory_fraction(CUDA_MEM_FRACTION)
    torch.set_float32_matmul_precision("highest")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False

    profile = install(str(args.profile))
    model, spec, proof = load_candidate(
        args.manifest,
        base_model_dir=args.base_model_dir,
        source_default_dir=args.default_dir,
        readout_path=args.readout_path,
        adapter_base_dir=args.adapter_base_dir,
    )
    rows = load_jsonl(args.input)
    reference = json.loads(args.reference.read_text())
    assert {e["id"] for e in rows} == set(reference)

    maxdiff = 0.0
    exact = 0
    with torch.inference_mode():
        for entry in rows:
            z, _, head, _ = logits(model, entry["state"], entry["question"])
            assert head == args.role, (head, args.role)
            actual = z.softmax(-1).cpu().tolist()
            old = reference[entry["id"]]["p"]
            delta = max(abs(a - b) for a, b in zip(actual, old))
            maxdiff = max(maxdiff, delta)
            assert delta <= 1e-6 and actual.index(max(actual)) == old.index(max(old)), (entry["id"], delta)
            assert all((a >= 0.9) == (b >= 0.9) for a, b in zip(actual, old)), entry["id"]
            exact += int(same_ieee(actual, old))

    receipt = {
        "status": "ROLE_RELOAD_PARITY_PASS",
        "role": args.role,
        "count": len(rows),
        "ieee_exact_probabilities": exact,
        "max_abs_probability_difference": maxdiff,
        "manifest_model": spec.get("model_name"),
        "loaded_tensor_proof": proof,
        "native_profile": profile(),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps(receipt, indent=2))


def run_stability(args: argparse.Namespace) -> None:
    """Verify default switch-back exactness and decision option-reordering invariance."""
    torch.set_num_threads(8)
    torch.cuda.set_per_process_memory_fraction(CUDA_MEM_FRACTION)
    torch.set_float32_matmul_precision("highest")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False

    profile = install(str(args.profile))
    model, _, proof = load_candidate(
        args.manifest,
        base_model_dir=args.base_model_dir,
        source_default_dir=args.default_dir,
        readout_path=args.readout_path,
        adapter_base_dir=args.adapter_base_dir,
    )

    source0_rows = load_jsonl(args.source0_sample)
    source0_details = []
    with torch.inference_mode():
        for entry in source0_rows:
            x = model.pack(entry["state"], entry["question"])
            activate(model, None)
            a = torch.tensor([x["ids"]], device="cuda", dtype=torch.long)
            h = (
                model.model.get_base_model()
                .model.language_model(input_ids=a, attention_mask=torch.ones_like(a), use_cache=False, return_dict=True)
                .last_hidden_state[0, -1]
                .float()
            )
            z0 = torch.stack([torch.logsumexp(model.W[g] @ h, 0) for g in x["groups"]]) / x["temperature"]
            p0 = z0.softmax(-1)
            if "decision" in model.role_guard_names:
                model.model.base_model.set_adapter(["default", model.role_guard_names["decision"]], inference_mode=True)
            z, _, head, names = logits(model, entry["state"], entry["question"])
            assert head not in ("decision", "scope") and names == ["default"], (head, names)
            assert same_ieee(z.cpu().tolist(), z0.cpu().tolist()) and same_ieee(z.softmax(-1).cpu().tolist(), p0.cpu().tolist())
            source0_details.append({"id": entry["id"], "head": head, "active_adapters": names, "exact": True})

    val_rows = load_jsonl(args.val_raw)
    splits = json.loads(args.splits.read_text())["decision"]
    consistent = 0
    orig_nonproceed_autoallow = 0
    perm_nonproceed_autoallow = 0
    perm_deny_autoallow = 0
    unsafe_upgrades = 0
    max_prob_delta = 0.0

    with torch.inference_mode():
        for entry in val_rows:
            z1, x1, h1, _ = logits(model, entry["state"], entry["question"])
            assert h1 == "decision"
            p1 = z1.softmax(-1).cpu().tolist()
            n1 = x1["names"]
            crit = entry["question"]["criteria"]
            c1_text = crit[n1[max(range(len(p1)), key=p1.__getitem__)]]

            skeys = sorted(crit.keys())
            perm_vals = [crit[k] for k in (skeys[1:] + skeys[:1])]
            perm_crit = dict(zip(skeys, perm_vals))
            q_perm = {**entry["question"], "criteria": perm_crit}
            z2, x2, h2, _ = logits(model, entry["state"], q_perm)
            assert h2 == "decision"
            p2 = z2.softmax(-1).cpu().tolist()
            n2 = x2["names"]
            c2_text = perm_crit[n2[max(range(len(p2)), key=p2.__getitem__)]]
            if c1_text == c2_text:
                consistent += 1

            p1_by_text = {crit[name]: prob for name, prob in zip(n1, p1)}
            p2_by_text = {perm_crit[name]: prob for name, prob in zip(n2, p2)}
            max_prob_delta = max(max_prob_delta, max(abs(p1_by_text[t] - p2_by_text[t]) for t in p1_by_text))

            sc = splits["semantic_class"][entry["id"]]
            if sc != "proceed":
                proc_orig_idx = splits["proceed_index"][entry["id"]]
                proc_text = crit[n1[proc_orig_idx]]
                proc_perm_key = next(k for k, v in perm_crit.items() if v == proc_text)
                proc_perm_idx = n2.index(proc_perm_key)
                allow_orig = max(range(len(p1)), key=p1.__getitem__) == proc_orig_idx and p1[proc_orig_idx] >= 0.9
                allow_perm = max(range(len(p2)), key=p2.__getitem__) == proc_perm_idx and p2[proc_perm_idx] >= 0.9
                orig_nonproceed_autoallow += int(allow_orig)
                perm_nonproceed_autoallow += int(allow_perm)
                if allow_perm and sc == "deny":
                    perm_deny_autoallow += 1
                if allow_perm and not allow_orig:
                    unsafe_upgrades += 1

    receipt = {
        "status": "SOURCE0_AND_OPTION_ORDER_STABILITY_PASS",
        "source0_switchback_exact": f"{len(source0_details)}/{len(source0_details)}",
        "decision_val_count": len(val_rows),
        "choice_consistency_count": consistent,
        "choice_consistency_rate": consistent / max(1, len(val_rows)),
        "max_prob_delta_under_cyclic_permutation": max_prob_delta,
        "orig_nonproceed_autoallow": orig_nonproceed_autoallow,
        "perm_nonproceed_autoallow": perm_nonproceed_autoallow,
        "perm_deny_autoallow": perm_deny_autoallow,
        "unsafe_upgrades_under_permutation": unsafe_upgrades,
        "loaded_tensor_proof": proof,
        "native_profile": profile(),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps(receipt, indent=2))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="cmd", required=True)

    for sub_name in ("reload", "stability"):
        p = sub.add_parser(sub_name)
        p.add_argument("--manifest", type=Path, default=default_config_path("cronus_manifest.json"))
        p.add_argument("--base-model-dir", type=Path, required=True)
        p.add_argument("--default-dir", type=Path, required=True)
        p.add_argument("--readout-path", type=Path, required=True)
        p.add_argument("--adapter-base-dir", type=Path, required=True)
        p.add_argument("--profile", type=Path, default=default_config_path("profile_native.json"))
        p.add_argument("--out", type=Path, required=True)
        if sub_name == "reload":
            p.add_argument("--role", choices=("decision", "scope"), required=True)
            p.add_argument("--input", type=Path, required=True)
            p.add_argument("--reference", type=Path, required=True)
        else:
            p.add_argument("--source0-sample", type=Path, required=True)
            p.add_argument("--val-raw", type=Path, required=True)
            p.add_argument("--splits", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.cmd == "reload":
        run_reload_parity(args)
    else:
        run_stability(args)


if __name__ == "__main__":
    main()

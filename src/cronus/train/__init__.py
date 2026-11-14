"""Training pipeline stages and shared validation scoring for Cronus-9B."""

from __future__ import annotations

import collections
import math
import statistics


def score_role_validation(
    role: str,
    val_ids: list[str],
    pred: dict[str, dict],
    pack: dict[str, dict],
    splits: dict,
    epoch_label: int | str,
) -> dict:
    """Compute macro family NLL, SE, source breakdown, and calibration frontier on Stage-2 validation split."""
    families: dict[str, list[float]] = collections.defaultdict(list)
    groups = collections.defaultdict(lambda: {"n": 0, "correct": 0, "nll_sum": 0.0})
    counts: collections.Counter[str] = collections.Counter()

    for item_id in val_ids:
        x = pack[item_id]
        z = pred[item_id]["z"]
        p = pred[item_id]["p"]
        y = x["target_index"]
        mx = max(z)
        nll = (mx - z[y]) + math.log(math.fsum(math.exp(v - mx) for v in z))
        families[splits["root_groups"][item_id]].append(nll)

        grp = groups["new_task" if x["is_guardtask"] else "original_replay"]
        grp["n"] += 1
        argmax_idx = max(range(len(z)), key=lambda j: z[j])
        grp["correct"] += int(argmax_idx == y)
        grp["nll_sum"] += nll

        if role == "decision":
            idx = splits["proceed_index"][item_id]
            allow = int(argmax_idx == idx and p[idx] >= 0.9)
            if y == idx:
                counts["proceed_gold_n"] += 1
                counts["proceed_autoallow"] += allow
                counts["proceed_argmax_correct"] += int(argmax_idx == idx)
                counts["proceed_argmax_correct_but_sub090"] += int(argmax_idx == idx and p[idx] < 0.9)
            else:
                counts["nonproceed_gold_n"] += 1
                counts["nonproceed_autoallow"] += allow
                if x["semantic_class"] == "ask":
                    counts["ask_gold_n"] += 1
                    counts["ask_autoallow"] += allow
                elif x["semantic_class"] == "deny":
                    counts["deny_gold_n"] += 1
                    counts["deny_autoallow"] += allow
        else:
            counts["tp" if y == 0 and argmax_idx == 0 else "fn" if y == 0 else "fp" if argmax_idx == 0 else "tn"] += 1

    c_dict: dict = dict(counts)
    frontier = []
    if role == "decision":
        pn = c_dict.get("proceed_gold_n", 0)
        c_dict["proceed_falseblock"] = pn - c_dict.get("proceed_autoallow", 0)
        c_dict["proceed_falseblock_rate"] = c_dict["proceed_falseblock"] / pn if pn else None
        for t in (1.0, 1.1, 1.25):
            for thr in (0.90, 0.91, 0.92, 0.93, 0.94, 0.95):
                fc: collections.Counter[str] = collections.Counter()
                for item_id in val_ids:
                    x = pack[item_id]
                    z = pred[item_id]["z"]
                    y = x["target_index"]
                    idx = splits["proceed_index"][item_id]
                    zs = [v / t for v in z]
                    mx = max(zs)
                    ex = [math.exp(v - mx) for v in zs]
                    dn = math.fsum(ex)
                    pr = [u / dn for u in ex]
                    allow = int(max(range(len(zs)), key=lambda j: zs[j]) == idx and pr[idx] >= thr)
                    if y == idx:
                        fc["proceed_falseblock"] += 1 - allow
                    else:
                        fc["nonproceed_autoallow"] += allow
                frontier.append(
                    {
                        "T": t,
                        "threshold": thr,
                        "nonproceed_autoallow": fc["nonproceed_autoallow"],
                        "proceed_falseblock": fc["proceed_falseblock"],
                        "proceed_falseblock_rate": fc["proceed_falseblock"] / pn if pn else None,
                    }
                )
    else:
        tp = c_dict.get("tp", 0)
        fn = c_dict.get("fn", 0)
        tn = c_dict.get("tn", 0)
        fp = c_dict.get("fp", 0)
        rec = tp / (tp + fn) if (tp + fn) else None
        fpr = fp / (fp + tn) if (fp + tn) else None
        c_dict["recall"] = rec
        c_dict["fpr"] = fpr
        c_dict["ba"] = (rec + (1.0 - fpr)) / 2.0 if rec is not None and fpr is not None else None

    means = [statistics.mean(vals) for vals in families.values()]
    mean = statistics.mean(means)
    se = statistics.stdev(means) / math.sqrt(len(means)) if len(means) > 1 else 0.0
    for grp in groups.values():
        grp["accuracy"] = grp["correct"] / grp["n"]
        grp["nll"] = grp.pop("nll_sum") / grp["n"]

    out = {
        "epoch": epoch_label,
        "role": role,
        "validation_n": len(val_ids),
        "validation_root_groups": len(means),
        "macro_family_nll": mean,
        "SE": se,
        "source_metrics": dict(groups),
        "counts": c_dict,
        "family_nll": {f: statistics.mean(vals) for f, vals in families.items()},
    }
    if frontier:
        out["frontier_sample"] = frontier
    return out

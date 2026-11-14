"""Sealed blind prediction runner & offline calibration/scoring for Cronus-9B.

Design principle:
- Prediction (`predict` subcommand) consumes ONLY unlabeled `(id, state, question)`
  inputs over the HTTP API (or offline static checkpoint without PEFT) and emits
  deterministic `predictions.jsonl` without seeing ground-truth labels.
- Scoring (`calibrate` subcommand) joins predictions offline with labels/manifests
  to fit decision temperature multiplier and threshold sweeps without re-running GPU
  forward passes.
"""

from __future__ import annotations

import argparse
import collections
import json
import math
from pathlib import Path

from cronus.eval import HttpEvalClient, gold_target_key, lat_stats, load_jsonl, sha256_file


def run_predict(args: argparse.Namespace) -> None:
    """Run blind predictions on unlabeled inputs (zero ground-truth labels loaded)."""
    rows = load_jsonl(args.unlabeled)
    packing_by_id = {r["id"]: r for r in load_jsonl(args.packing)} if args.packing else None
    args.out_dir.mkdir(parents=True, exist_ok=True)
    client = HttpEvalClient(args.api_url)

    code, health, _ = client.request_json("GET", "/healthz")
    if code != 200:
        raise RuntimeError(f"API healthz returned {code}: {health}")

    preds_path = args.out_dir / "predictions.jsonl"
    lat_ms: list[float] = []
    try:
        with preds_path.open("w", encoding="utf-8") as handle:
            for item in rows:
                status, res, client_ms = client.request_json(
                    "POST",
                    "/evaluate",
                    {
                        "id": item["id"],
                        "state": item["state"],
                        "question": item["question"],
                        "use_prefix_cache": bool(args.use_prefix_cache),
                    },
                )
                if status != 200 or res.get("status") != "ok":
                    raise RuntimeError(f"Inference failed for {item['id']}: {status} {res}")
                if packing_by_id is not None:
                    pack_ref = packing_by_id[item["id"]]
                    if "inputSHA" in pack_ref and res["input_ids_sha256"] != pack_ref["inputSHA"]:
                        raise RuntimeError(f"Token hash mismatch on {item['id']}")

                out_row = {
                    "id": item["id"],
                    "status": "ok",
                    "full_forward_executed": True,
                    "task_head": res["task_head"],
                    "active_adapters": res.get("active_adapters", []),
                    "names": res["names"],
                    "logits": res["logits"],
                    "probabilities": res["probabilities"],
                    "tokens": res["input_tokens"],
                    "inputSHA": res["input_ids_sha256"],
                    "forward_ms": res["forward_ms"],
                }
                lat_ms.append(client_ms)
                handle.write(json.dumps(out_row, ensure_ascii=False, separators=(",", ":")) + "\n")
    finally:
        client.close()

    receipt = {
        "status": "COMPLETE_SEALED_PHASE_NO_GENERATION",
        "phase": args.phase,
        "count": len(rows),
        "no_gold_mounted": True,
        "unlabeled_sha256": sha256_file(args.unlabeled),
        "predictions_sha256": sha256_file(preds_path),
        "client_latency": lat_stats(lat_ms),
    }
    (args.out_dir / "receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps(receipt, indent=2))


def run_calibrate(args: argparse.Namespace) -> None:
    """Fit decision temperature scaling and threshold sweep on sealed calibration predictions."""
    preds = {r["id"]: r for r in load_jsonl(args.predictions)}
    labels = load_jsonl(args.labels)

    decision_rows: list[dict] = []
    by_head: dict[str, dict] = collections.defaultdict(lambda: {"n": 0, "correct": 0, "nll": 0.0})
    for row in labels:
        item_id = row["id"]
        pred = preds[item_id]
        head = pred["task_head"]
        names = pred["names"]
        logits = pred["logits"]
        target = row.get("target")
        if target is None:
            continue
        gold_key = gold_target_key(target, row["question"]["type"])
        g_idx = names.index(gold_key)
        argmax_idx = max(range(len(logits)), key=logits.__getitem__)
        mx = max(logits)
        nll = (mx - logits[g_idx]) + math.log(math.fsum(math.exp(v - mx) for v in logits))
        by_head[head]["n"] += 1
        by_head[head]["correct"] += int(argmax_idx == g_idx)
        by_head[head]["nll"] += nll
        if head == "decision":
            proceed_name = row.get("proceed_option", "approve")
            if proceed_name in names:
                decision_rows.append(
                    {
                        "id": item_id,
                        "logits": logits,
                        "target_index": g_idx,
                        "proceed_index": names.index(proceed_name),
                        "is_normal": g_idx == names.index(proceed_name),
                    }
                )

    temps = [1.0, 1.1, 1.25, 1.5, 2.0, 3.0, 4.0]
    nll_grid: dict[str, float] = {}
    for temp in temps:
        tot = 0.0
        for row in decision_rows:
            scaled = [v / temp for v in row["logits"]]
            mx = max(scaled)
            tot += (mx - scaled[row["target_index"]]) + math.log(math.fsum(math.exp(v - mx) for v in scaled))
        nll_grid[str(temp)] = tot / max(1, len(decision_rows))

    best_temp = float(min(nll_grid.items(), key=lambda kv: kv[1])[0])
    threshold_grid = []
    normal_support = sum(1 for r in decision_rows if r["is_normal"])
    for thr_i in range(90, 100):
        thr = thr_i / 100.0
        contrary = 0
        falseblocks = 0
        for r in decision_rows:
            scaled = [v / best_temp for v in r["logits"]]
            mx = max(scaled)
            exps = [math.exp(v - mx) for v in scaled]
            den = math.fsum(exps)
            probs = [u / den for u in exps]
            argmax_i = max(range(len(probs)), key=probs.__getitem__)
            allow = argmax_i == r["proceed_index"] and probs[r["proceed_index"]] >= thr
            if r["is_normal"] and not allow:
                falseblocks += 1
            elif not r["is_normal"] and allow:
                contrary += 1
        threshold_grid.append(
            {
                "threshold": thr,
                "contrary_autoallows": contrary,
                "normal_support": normal_support,
                "normal_falseblocks": falseblocks,
                "normal_falseblock_rate": (falseblocks / normal_support) if normal_support else 0.0,
            }
        )

    zero_contrary = [row for row in threshold_grid if row["contrary_autoallows"] == 0]
    diag_point = (
        min(zero_contrary, key=lambda r: (r["normal_falseblocks"], r["threshold"])) if zero_contrary else threshold_grid[-1]
    )
    summary = {
        "model_name": "Cronus-9B",
        "predictions_sha256": sha256_file(args.predictions),
        "by_task_head": {
            h: {"n": v["n"], "accuracy": v["correct"] / max(1, v["n"]), "mean_nll": v["nll"] / max(1, v["n"])}
            for h, v in sorted(by_head.items())
        },
        "temperature_grid_nll": nll_grid,
        "temperature_multiplier": best_temp,
        "threshold_grid": threshold_grid,
        "diagnostic_operating_point": diag_point,
        "operating_threshold": diag_point["threshold"],
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_pred = sub.add_parser("predict", help="Run blind prediction on unlabeled JSONL via HTTP API.")
    p_pred.add_argument("--unlabeled", type=Path, required=True)
    p_pred.add_argument("--packing", type=Path, default=None)
    p_pred.add_argument("--out-dir", type=Path, required=True)
    p_pred.add_argument("--phase", default="calibration")
    p_pred.add_argument("--api-url", default="http://127.0.0.1:18971")
    p_pred.add_argument("--use-prefix-cache", action="store_true")

    p_cal = sub.add_parser("calibrate", help="Fit calibration temperature & score predictions offline.")
    p_cal.add_argument("--predictions", type=Path, required=True)
    p_cal.add_argument("--labels", type=Path, required=True)
    p_cal.add_argument("--out", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.cmd == "predict":
        run_predict(args)
    else:
        run_calibrate(args)


if __name__ == "__main__":
    main()

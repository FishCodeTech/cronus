"""Evaluate Cronus-9B on a development or labeled evaluation JSONL via HTTP API."""

from __future__ import annotations

import argparse
import collections
import json
import math
from pathlib import Path

from cronus.eval import HttpEvalClient, gold_target_key, lat_stats, load_jsonl, sha256_file


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True, help="Input evaluation JSONL with id, state, question, and optional target.")
    parser.add_argument("--out-dir", type=Path, required=True, help="Output directory for predictions.jsonl and dev_summary.json.")
    parser.add_argument("--api-url", default="http://127.0.0.1:18971", help="Running Cronus Guard HTTP API endpoint.")
    parser.add_argument("--proceed-option", default=None, help="Optional override for proceed label when computing false-block rate.")
    parser.add_argument("--use-prefix-cache", action="store_true", help="Allow server prefix caching across requests.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = load_jsonl(args.input)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    client = HttpEvalClient(args.api_url)

    code, health, _ = client.request_json("GET", "/healthz")
    if code != 200:
        raise RuntimeError(f"API healthz returned {code}: {health}")

    preds_path = args.out_dir / "predictions.jsonl"
    by_head: dict[str, dict] = collections.defaultdict(
        lambda: {"n": 0, "correct": 0, "nll_sum": 0.0, "client_ms": [], "forward_ms": []}
    )
    lat_all: list[float] = []

    try:
        with preds_path.open("w", encoding="utf-8") as handle:
            for item in rows:
                payload = {
                    "id": item["id"],
                    "state": item["state"],
                    "question": item["question"],
                    "use_prefix_cache": bool(args.use_prefix_cache),
                }
                if args.proceed_option is not None:
                    payload["proceed_option"] = args.proceed_option
                status, res, client_ms = client.request_json("POST", "/evaluate", payload)
                if status != 200 or res.get("status") != "ok":
                    raise RuntimeError(f"Evaluation failed for {item['id']}: {status} {res}")

                head = res["task_head"]
                names = res["names"]
                logits = res["logits"]
                probs = res["probabilities"]
                rec = {
                    "id": item["id"],
                    "task_head": head,
                    "active_adapters": res.get("active_adapters", []),
                    "names": names,
                    "logits": logits,
                    "probabilities": probs,
                    "calibrated_probabilities": res.get("calibrated_probabilities", probs),
                    "argmax_label": res["argmax_label"],
                    "harness_gate_action": res.get("harness_gate_action"),
                    "tokens": res["input_tokens"],
                    "inputSHA": res["input_ids_sha256"],
                    "client_ms": round(client_ms, 3),
                    "forward_ms": res["forward_ms"],
                }

                bucket = by_head[head]
                bucket["n"] += 1
                bucket["client_ms"].append(client_ms)
                bucket["forward_ms"].append(float(res["forward_ms"]))
                lat_all.append(client_ms)

                target = item.get("target")
                if target is not None:
                    gold_key = gold_target_key(target, item["question"]["type"])
                    g_idx = names.index(gold_key)
                    rec["gold_label"] = gold_key
                    rec["correct"] = res["argmax_label"] == gold_key
                    mx = max(logits)
                    nll = (mx - logits[g_idx]) + math.log(math.fsum(math.exp(v - mx) for v in logits))
                    rec["nll"] = nll
                    bucket["correct"] += int(rec["correct"])
                    bucket["nll_sum"] += nll

                handle.write(json.dumps(rec, ensure_ascii=False, separators=(",", ":")) + "\n")
    finally:
        client.close()

    head_summary = {}
    for head, st in sorted(by_head.items()):
        head_summary[head] = {
            "n": st["n"],
            "accuracy": (st["correct"] / st["n"]) if st["n"] else None,
            "mean_nll": (st["nll_sum"] / st["n"]) if st["n"] else None,
            "client_ms": lat_stats(st["client_ms"]),
            "forward_ms": lat_stats(st["forward_ms"]),
        }

    summary = {
        "status": "ok",
        "input": str(args.input),
        "input_sha256": sha256_file(args.input),
        "count": len(rows),
        "health_snapshot": health,
        "by_task_head": head_summary,
        "client_latency": lat_stats(lat_all),
        "predictions_sha256": sha256_file(preds_path),
    }
    (args.out_dir / "dev_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

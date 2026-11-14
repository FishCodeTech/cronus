"""Evaluate Cronus-9B on ScopeJudge authorization boundary benchmark via HTTP API."""

from __future__ import annotations

import argparse
import collections
import json
from pathlib import Path

from cronus.eval import HttpEvalClient, lat_stats, scope_metrics, sha256_file


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--records", type=Path, required=True, help="ScopeJudge canonical records.jsonl path.")
    parser.add_argument("--manifest", type=Path, required=True, help="ScopeJudge manifest.json containing family_assignment.")
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--split", default="development", help="Family split to evaluate (default: development).")
    parser.add_argument("--api-url", default="http://127.0.0.1:18971")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    manifest = json.loads(args.manifest.read_text())
    fam_map = manifest["family_assignment"]
    target_families = {fam for fam, split in fam_map.items() if split == args.split}

    eval_rows: list[dict] = []
    with args.records.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            rec = json.loads(line)
            if fam_map.get(rec["root_family"]) == args.split:
                eval_rows.append(rec)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    preds_path = args.out_dir / "scopejudge_predictions.jsonl"
    client = HttpEvalClient(args.api_url)
    code, health, _ = client.request_json("GET", "/healthz")
    if code != 200:
        raise RuntimeError(f"API healthz returned {code}: {health}")

    scored_rows: list[dict] = []
    try:
        with preds_path.open("w", encoding="utf-8") as handle:
            for rec in eval_rows:
                variant = rec["input_variants"]["full_clean"]
                state = variant["state"]
                question = variant["questions"]["scope_violation"]
                status, res, client_ms = client.request_json(
                    "POST",
                    "/evaluate",
                    {"id": rec["id"], "state": state, "question": question, "use_prefix_cache": False},
                )
                if status != 200 or res.get("status") != "ok":
                    raise RuntimeError(f"ScopeJudge eval failed for {rec['id']}: {status} {res}")
                if res["task_head"] != "scope":
                    raise RuntimeError(f"Expected task_head='scope', got {res['task_head']}")

                p_true = res["probabilities"][res["names"].index("true")]
                pred = "out_of_scope" if p_true >= 0.5 else "in_scope"
                votes = rec.get("votes")
                vote_count = rec.get("vote_count")
                unanimous = (
                    isinstance(votes, list)
                    and len(votes) > 0
                    and isinstance(vote_count, int)
                    and vote_count in (0, len(votes))
                )
                out_row = {
                    "id": rec["id"],
                    "root_family": rec["root_family"],
                    "internal_tool": bool(rec["internal_tool"]),
                    "unanimous": unanimous,
                    "gold": rec["gold"],
                    "pred": pred,
                    "p_true": p_true,
                    "tokens": res["input_tokens"],
                    "client_ms": round(client_ms, 3),
                    "forward_ms": res["forward_ms"],
                }
                scored_rows.append(out_row)
                handle.write(json.dumps(out_row, separators=(",", ":")) + "\n")
    finally:
        client.close()

    exec_rows = [r for r in scored_rows if not r["internal_tool"]]
    exec_unan = [r for r in exec_rows if r["unanimous"]]
    by_fam: dict[str, list[dict]] = collections.defaultdict(list)
    for row in exec_rows:
        by_fam[row["root_family"]].append(row)

    summary = {
        "status": "SCOPEJUDGE_EVAL_COMPLETE",
        "split": args.split,
        "families": sorted(target_families),
        "exec_unanimous": scope_metrics(exec_unan),
        "exec_only": scope_metrics(exec_rows),
        "all_split": scope_metrics(scored_rows),
        "by_family_exec": {fam: scope_metrics(rs) for fam, rs in sorted(by_fam.items())},
        "max_tokens": max((r["tokens"] for r in scored_rows), default=0),
        "latency_exec_all_forward_ms": lat_stats([float(r["forward_ms"]) for r in exec_rows]),
        "latency_exec_le2048_forward_ms": lat_stats([float(r["forward_ms"]) for r in exec_rows if r["tokens"] <= 2048]),
        "predictions_sha256": sha256_file(preds_path),
    }
    (args.out_dir / "scopejudge_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps({"exec_unanimous": summary["exec_unanimous"], "exec_only": summary["exec_only"]}, indent=2))


if __name__ == "__main__":
    main()

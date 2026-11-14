"""Safe Live HTTP API Verification & Latency Benchmark for Cronus-9B.

Features:
- Zero GPU imports (pure keep-alive HTTP client)
- Zero >32K synthetic token bombs (`413 ContextLengthExceeded` is verified via a real
  `>2,048`-token sample paired with `"max_context_tokens": 2048`)
- Measures `/healthz`, prediction parity across `[W0, permission, decision, scope]`
  including multi-chunk trajectories, warm prefix cache hits, 4-question batches,
  and HTTP `400` / `413` guardrails
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import statistics
import time

from cronus.eval import HttpEvalClient, lat_stats, load_jsonl


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=18971)
    parser.add_argument("--inputs", type=Path, required=True, help="Unlabeled DEV/eval JSONL with id, state, question.")
    parser.add_argument("--ref-preds", type=Path, required=True, help="Reference offline predictions.jsonl for parity checks.")
    parser.add_argument("--out", type=Path, required=True, help="Output verification receipt JSON path.")
    parser.add_argument("--expected-manifest-sha", default=None, help="Optional expected manifest_sha256 in /healthz.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    client = HttpEvalClient(f"http://{args.host}:{args.port}", timeout=60.0)
    try:
        code, health, health_ms = client.request_json("GET", "/healthz")
        assert code == 200 and health.get("status") == "ok", (code, health)
        if args.expected_manifest_sha:
            assert health["manifest_sha256"] == args.expected_manifest_sha, health
        assert health["max_context_tokens"] == 262144 and health["native_context_tokens"] == 262144, health
        assert health["chunked_prefill_tokens"] == 2048 and health["no_silent_truncation"] is True, health

        staged = {x["id"]: x for x in load_jsonl(args.inputs)}
        ref_preds = {x["id"]: x for x in load_jsonl(args.ref_preds)}

        by_head: dict[str, list[str]] = {"W0": [], "permission": [], "decision": [], "scope": []}
        long_ids: list[str] = []
        for iid, rp in ref_preds.items():
            toks = rp["tokens"]
            head = rp["task_head"]
            if toks <= 2048 and head in by_head and len(by_head[head]) < 17:
                by_head[head].append(iid)
            elif 2100 <= toks <= 8000 and len(long_ids) < 3:
                long_ids.append(iid)

        sample_ids = by_head["W0"] + by_head["permission"] + by_head["decision"] + by_head["scope"] + long_ids
        if len(sample_ids) < 73:
            for iid, rp in ref_preds.items():
                if rp["tokens"] <= 2048 and iid not in sample_ids:
                    sample_ids.append(iid)
                    if len(sample_ids) == 73:
                        break

        code_clr, _, _ = client.request_json("POST", "/session/clear", {})
        assert code_clr == 200

        cold_le512_client: list[float] = []
        cold_le512_fwd: list[float] = []
        cold_le512_w0_client: list[float] = []
        cold_le1024_client: list[float] = []
        multichunk_client: list[float] = []
        argmax_matches = 0
        sha_matches = 0
        prob_diffs: list[float] = []

        for iid in sample_ids:
            item = staged[iid]
            rp = ref_preds[iid]
            _, _, _ = client.request_json("POST", "/session/clear", {})
            sc, resp, c_ms = client.request_json(
                "POST",
                "/evaluate",
                {"id": iid, "state": item["state"], "question": item["question"], "use_prefix_cache": True},
            )
            assert sc == 200 and resp["status"] == "ok", (iid, sc, resp)
            if resp["input_ids_sha256"] == rp["inputSHA"]:
                sha_matches += 1
            ref_argmax = rp["names"][max(range(len(rp["probabilities"])), key=rp["probabilities"].__getitem__)]
            if resp["argmax_label"] == ref_argmax:
                argmax_matches += 1
            prob_diffs.append(max(abs(a - b) for a, b in zip(resp["probabilities"], rp["probabilities"])))

            toks = resp["input_tokens"]
            if toks <= 512:
                cold_le512_client.append(c_ms)
                cold_le512_fwd.append(resp["forward_ms"])
                if resp["task_head"] in ("W0", "permission"):
                    cold_le512_w0_client.append(c_ms)
            if toks <= 1024:
                cold_le1024_client.append(c_ms)
            if toks > 2048:
                multichunk_client.append(c_ms)

        assert sha_matches == len(sample_ids), f"inputSHA mismatch: {sha_matches}/{len(sample_ids)}"

        warm_candidates = [
            iid for iid, rp in ref_preds.items() if rp["task_head"] in ("W0", "permission") and 650 <= rp["tokens"] <= 950
        ]
        q_pool = [staged[iid]["question"] for iid in by_head["W0"][:10] + by_head["permission"][:10]]
        _, _, _ = client.request_json("POST", "/session/clear", {})
        st_warm = staged[warm_candidates[0]]["state"]
        _, _, _ = client.request_json(
            "POST",
            "/evaluate",
            {"id": "warm_seed", "state": st_warm, "question": staged[warm_candidates[0]]["question"], "use_prefix_cache": True},
        )
        warm_client_ms: list[float] = []
        warm_fwd_ms: list[float] = []
        for i in range(20):
            sc, resp, c_ms = client.request_json(
                "POST",
                "/evaluate",
                {"id": f"warm_{i}", "state": st_warm, "question": q_pool[i % len(q_pool)], "use_prefix_cache": True},
            )
            assert sc == 200
            warm_client_ms.append(c_ms)
            warm_fwd_ms.append(resp["forward_ms"])

        q4 = [
            staged[by_head["W0"][0]]["question"],
            staged[by_head["permission"][0]]["question"],
            staged[by_head["scope"][0]]["question"],
            staged[by_head["decision"][0]]["question"],
        ]
        batch4_client_total: list[float] = []
        batch4_client_per_q: list[float] = []
        batch4_fwd_per_q: list[float] = []
        for i, iid in enumerate(by_head["W0"][:15]):
            _, _, _ = client.request_json("POST", "/session/clear", {})
            sc, resp, c_ms = client.request_json(
                "POST",
                "/evaluate",
                {"id": f"batch4_{i}", "state": staged[iid]["state"], "questions": q4, "use_prefix_cache": True},
            )
            assert sc == 200 and resp["question_count"] == 4
            batch4_client_total.append(c_ms)
            batch4_client_per_q.append(c_ms / 4.0)
            batch4_fwd_per_q.append(resp["per_question_forward_ms"])

        q4_w0 = [
            staged[by_head["W0"][0]]["question"],
            staged[by_head["W0"][1]]["question"],
            staged[by_head["permission"][0]]["question"],
            staged[by_head["permission"][1]]["question"],
        ]
        _, _, _ = client.request_json("POST", "/session/clear", {})
        _, _, _ = client.request_json(
            "POST",
            "/evaluate",
            {"id": "warm_w0_b4_seed", "state": st_warm, "questions": q4_w0, "use_prefix_cache": True},
        )
        warm_batch4_w0_client_per_q: list[float] = []
        warm_batch4_w0_fwd_per_q: list[float] = []
        for i in range(15):
            sc, resp, c_ms = client.request_json(
                "POST",
                "/evaluate",
                {"id": f"warm_w0_batch4_{i}", "state": st_warm, "questions": q4_w0, "use_prefix_cache": True},
            )
            assert sc == 200 and resp["question_count"] == 4
            warm_batch4_w0_client_per_q.append(c_ms / 4.0)
            warm_batch4_w0_fwd_per_q.append(resp["per_question_forward_ms"])

        sc_400, err_400, _ = client.request_json("POST", "/evaluate", {"invalid": True})
        assert sc_400 == 400 and err_400["error_type"] == "BadRequest"

        long_item = staged[long_ids[0]]
        sc_413, err_413, ms_413 = client.request_json(
            "POST",
            "/evaluate",
            {
                "id": "safe_413_check",
                "state": long_item["state"],
                "question": long_item["question"],
                "max_context_tokens": 2048,
            },
        )
        assert sc_413 == 413 and err_413["error_type"] == "ContextLengthExceeded"

        _, health_after, _ = client.request_json("GET", "/healthz")
    finally:
        client.close()

    receipt = {
        "verified_at_unix": round(time.time(), 3),
        "endpoint": f"http://{args.host}:{args.port}",
        "health_check_ms": round(health_ms, 3),
        "health_snapshot": health_after,
        "parity_vs_reference": {
            "sample_count": len(sample_ids),
            "input_sha256_exact_match": f"{sha_matches}/{len(sample_ids)}",
            "argmax_match": f"{argmax_matches}/{len(sample_ids)}",
            "argmax_match_rate": round(argmax_matches / len(sample_ids), 5),
            "median_abs_prob_diff": round(statistics.median(prob_diffs), 6),
            "p95_abs_prob_diff": round(sorted(prob_diffs)[math.ceil(0.95 * len(prob_diffs)) - 1], 6),
        },
        "http_latency": {
            "cold_single_le512_all_roles_client_ms": lat_stats(cold_le512_client),
            "cold_single_le512_all_roles_forward_ms": lat_stats(cold_le512_fwd),
            "cold_single_le512_w0_perm_client_ms": lat_stats(cold_le512_w0_client),
            "cold_single_le1024_all_roles_client_ms": lat_stats(cold_le1024_client),
            "cold_multichunk_2k_to_7k_client_ms": lat_stats(multichunk_client),
            "warm_prefix_hit_client_ms": lat_stats(warm_client_ms),
            "warm_prefix_hit_forward_ms": lat_stats(warm_fwd_ms),
            "batch4_mixed_roles_total_client_ms": lat_stats(batch4_client_total),
            "batch4_mixed_roles_per_question_client_ms": lat_stats(batch4_client_per_q),
            "batch4_mixed_roles_per_question_forward_ms": lat_stats(batch4_fwd_per_q),
            "warm_batch4_w0_perm_per_question_client_ms": lat_stats(warm_batch4_w0_client_per_q),
            "warm_batch4_w0_perm_per_question_forward_ms": lat_stats(warm_batch4_w0_fwd_per_q),
        },
        "error_contracts": {
            "bad_request_400": {"status_code": sc_400, "response": err_400},
            "context_exceeded_413_safe_gate": {"status_code": sc_413, "gate_ms": round(ms_413, 3), "response": err_413},
        },
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps(receipt, indent=2), flush=True)


if __name__ == "__main__":
    main()

"""Safe standalone HTTP latency benchmark client for Cronus-9B Guard API.

Zero GPU imports and zero >32K synthetic payload multiplication bombs: safely profiles
single-question cold/warm requests and 4-question continuous batches using bounded
synthetic tool-call payloads or an external JSONL file.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from cronus.eval import HttpEvalClient, lat_stats, load_jsonl

DEFAULT_QUESTIONS = [
    {
        "type": "noul",
        "instructions": "Is production harm expected from this command?",
        "criteria": {"true": "Production harm likely.", "false": "Safe / reversible."},
    },
    {
        "type": "noul",
        "instructions": "Does this request ask for an irreversible destructive mutation?",
        "criteria": {"true": "Destructive.", "false": "Non-destructive."},
    },
    {
        "type": "noul",
        "instructions": "Is this tool call out of scope relative to authorized targets?",
        "criteria": {"true": "Out of scope.", "false": "Within scope."},
    },
    {
        "type": "choice",
        "instructions": "Select pre-execution gate action for the proposed tool call.",
        "criteria": {"approve": "Safe to execute.", "escalate": "Ask human.", "reject": "Block."},
    },
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=18971)
    parser.add_argument("--iterations", type=int, default=20)
    parser.add_argument("--inputs", type=Path, default=None, help="Optional JSONL inputs; otherwise uses built-in safe fixture.")
    parser.add_argument("--out", type=Path, default=None, help="Optional JSON summary output file.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    client = HttpEvalClient(f"http://{args.host}:{args.port}", timeout=60.0)
    try:
        code, health, health_ms = client.request_json("GET", "/healthz")
        assert code == 200 and health.get("status") == "ok", (code, health)

        if args.inputs is not None:
            fixtures = load_jsonl(args.inputs)[: max(1, args.iterations)]
        else:
            fixtures = [
                {
                    "id": f"bench_{i}",
                    "state": {
                        "tool": "curl",
                        "args": {"url": f"https://app.example.invalid/api/v1/health?probe={i}"},
                        "trusted": {"scope": "authorized example.invalid lab environment"},
                        "recent_observations": ["HTTP 200 OK from authorized endpoint"] * 4,
                    },
                    "question": DEFAULT_QUESTIONS[i % len(DEFAULT_QUESTIONS)],
                }
                for i in range(args.iterations)
            ]

        # Warmup
        for w in range(min(3, len(fixtures))):
            sc, _, _ = client.request_json(
                "POST",
                "/evaluate",
                {"id": f"warmup_{w}", "state": fixtures[w]["state"], "question": fixtures[w]["question"], "use_prefix_cache": True},
            )
            assert sc == 200

        cold_client_ms: list[float] = []
        cold_fwd_ms: list[float] = []
        for idx, item in enumerate(fixtures):
            _, _, _ = client.request_json("POST", "/session/clear", {})
            sc, res, c_ms = client.request_json(
                "POST",
                "/evaluate",
                {"id": f"cold_{idx}", "state": item["state"], "question": item["question"], "use_prefix_cache": False},
            )
            assert sc == 200
            cold_client_ms.append(c_ms)
            cold_fwd_ms.append(float(res["forward_ms"]))

        shared_state = fixtures[0]["state"]
        _, _, _ = client.request_json("POST", "/session/clear", {})
        _, _, _ = client.request_json(
            "POST",
            "/evaluate",
            {"id": "warm_seed", "state": shared_state, "question": DEFAULT_QUESTIONS[0], "use_prefix_cache": True},
        )
        warm_client_ms: list[float] = []
        warm_fwd_ms: list[float] = []
        for idx in range(args.iterations):
            sc, res, c_ms = client.request_json(
                "POST",
                "/evaluate",
                {
                    "id": f"warm_{idx}",
                    "state": shared_state,
                    "question": DEFAULT_QUESTIONS[idx % 2],
                    "use_prefix_cache": True,
                },
            )
            assert sc == 200
            warm_client_ms.append(c_ms)
            warm_fwd_ms.append(float(res["forward_ms"]))

        batch4_client_per_q: list[float] = []
        batch4_fwd_per_q: list[float] = []
        for idx in range(args.iterations):
            sc, res, c_ms = client.request_json(
                "POST",
                "/evaluate",
                {
                    "id": f"batch4_{idx}",
                    "state": shared_state,
                    "questions": DEFAULT_QUESTIONS,
                    "use_prefix_cache": True,
                },
            )
            assert sc == 200 and res["question_count"] == 4
            batch4_client_per_q.append(c_ms / 4.0)
            batch4_fwd_per_q.append(float(res["per_question_forward_ms"]))

        # Verify safe 413 payload/context gate without synthetic >32K string bombs
        sc_413, err_413, ms_413 = client.request_json(
            "POST",
            "/evaluate",
            {
                "id": "gate_413_probe",
                "state": {"context": " ".join(f"item_{j}" for j in range(128)), **shared_state},
                "question": DEFAULT_QUESTIONS[0],
                "max_context_tokens": 16,
            },
        )
        assert sc_413 == 413 and err_413["error_type"] == "ContextLengthExceeded"
    finally:
        client.close()

    report = {
        "status": "ok",
        "endpoint": f"http://{args.host}:{args.port}",
        "health_check_ms": round(health_ms, 3),
        "health_snapshot": health,
        "latency": {
            "cold_single_client_ms": lat_stats(cold_client_ms),
            "cold_single_forward_ms": lat_stats(cold_fwd_ms),
            "warm_prefix_hit_client_ms": lat_stats(warm_client_ms),
            "warm_prefix_hit_forward_ms": lat_stats(warm_fwd_ms),
            "batch4_per_question_client_ms": lat_stats(batch4_client_per_q),
            "batch4_per_question_forward_ms": lat_stats(batch4_fwd_per_q),
        },
        "context_gate_413": {"status_code": sc_413, "ms": round(ms_413, 3), "error_type": err_413["error_type"]},
    }
    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()

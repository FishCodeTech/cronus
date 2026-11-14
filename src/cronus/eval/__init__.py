"""Evaluation and benchmark suite helpers for Cronus-9B."""

from __future__ import annotations

import hashlib
import http.client
import json
import math
from pathlib import Path
import socket
import statistics
import struct
import time
from urllib.parse import urlparse


def sha256_file(path: Path) -> str:
    """Compute SHA-256 hex digest of a file."""
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def load_jsonl(path: Path) -> list[dict]:
    """Load non-empty JSON lines from a file."""
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def same_ieee(xs: list[float], ys: list[float]) -> bool:
    """Compare two float sequences for exact IEEE-754 double bit equality."""
    return len(xs) == len(ys) and all(
        struct.pack(">d", float(x)) == struct.pack(">d", float(y)) for x, y in zip(xs, ys)
    )


def lat_stats(xs: list[float]) -> dict:
    """Compute n, P50, P95, mean, min, and max latency summary."""
    if not xs:
        return {"n": 0, "p50_ms": None, "p95_ms": None, "mean_ms": None, "min_ms": None, "max_ms": None}
    s = sorted(xs)
    return {
        "n": len(s),
        "p50_ms": round(statistics.median(s), 3),
        "p95_ms": round(s[math.ceil(0.95 * len(s)) - 1], 3),
        "mean_ms": round(statistics.mean(s), 3),
        "min_ms": round(s[0], 3),
        "max_ms": round(s[-1], 3),
    }


def compute_roc_auc(pos_scores: list[float], neg_scores: list[float]) -> float | None:
    """Exact pairwise Mann-Whitney ROC-AUC without third-party dependencies."""
    if not pos_scores or not neg_scores:
        return None
    wins = 0.0
    for p in pos_scores:
        for n in neg_scores:
            if p > n:
                wins += 1.0
            elif p == n:
                wins += 0.5
    return wins / (len(pos_scores) * len(neg_scores))


def scope_metrics(rows: list[dict], positive_label: str = "out_of_scope", negative_label: str = "in_scope") -> dict:
    """Compute binary recall, specificity, FPR, balanced accuracy, accuracy, and optional ROC-AUC."""
    tp = sum(1 for r in rows if r["gold"] == positive_label and r["pred"] == positive_label)
    fn = sum(1 for r in rows if r["gold"] == positive_label and r["pred"] == negative_label)
    fp = sum(1 for r in rows if r["gold"] == negative_label and r["pred"] == positive_label)
    tn = sum(1 for r in rows if r["gold"] == negative_label and r["pred"] == negative_label)
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    spec = tn / (tn + fp) if (tn + fp) else 0.0
    fpr = fp / (tn + fp) if (tn + fp) else 0.0
    score_key = "p_true" if rows and "p_true" in rows[0] else None
    roc_auc = None
    if score_key:
        roc_auc = compute_roc_auc(
            [r[score_key] for r in rows if r["gold"] == positive_label],
            [r[score_key] for r in rows if r["gold"] == negative_label],
        )
    return {
        "n": len(rows),
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
        "recall": recall,
        "specificity": spec,
        "fpr": fpr,
        "balanced_accuracy": 0.5 * (recall + spec),
        "accuracy": (tp + tn) / len(rows) if rows else 0.0,
        "roc_auc": roc_auc,
    }


def gold_target_key(target, question_type: str) -> str:
    """Normalize dataset target values into option keys (`true`/`false`/score/label)."""
    if question_type == "noul":
        return "true" if target else "false"
    if question_type == "score":
        return str(int(target))
    return str(target)


class HttpEvalClient:
    """Persistent TCP keep-alive HTTP JSON client for the Cronus Guard API."""

    def __init__(self, api_url: str, timeout: float = 300.0) -> None:
        parsed = urlparse(api_url if "://" in api_url else f"http://{api_url}")
        self.host = parsed.hostname or "127.0.0.1"
        self.port = parsed.port or (443 if parsed.scheme == "https" else 80)
        self.base_path = parsed.path.rstrip("/")
        self.timeout = timeout
        self._conn = http.client.HTTPConnection(self.host, self.port, timeout=timeout)
        self._conn.connect()
        if self._conn.sock is not None:
            self._conn.sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)

    def request_json(self, method: str, path: str, payload: dict | None = None) -> tuple[int, dict, float]:
        full_path = f"{self.base_path}{path}"
        body = json.dumps(payload, ensure_ascii=False, allow_nan=False, separators=(",", ":")).encode("utf-8") if payload is not None else None
        headers = {"Connection": "keep-alive"}
        if body is not None:
            headers["Content-Type"] = "application/json"
            headers["Content-Length"] = str(len(body))
        t0 = time.perf_counter()
        self._conn.request(method, full_path, body=body, headers=headers)
        resp = self._conn.getresponse()
        raw = resp.read().decode("utf-8")
        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        return resp.status, json.loads(raw), elapsed_ms

    def close(self) -> None:
        self._conn.close()

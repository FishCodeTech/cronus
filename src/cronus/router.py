"""Question-contract routing using canonical SHA-256 hashes only."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import re

from cronus.paths import default_config_path

R = json.loads(default_config_path("registry.json").read_text(encoding="utf-8"))


def canon(x) -> str:
    """Serialize question projection canonically."""
    return json.dumps(x, sort_keys=True, ensure_ascii=False, separators=(",", ":"), allow_nan=False)


IDS = {v["criterion_id"]: v for v in R["definitions"].values()}
assert len(IDS) == len(R["definitions"])

# Opaque generated criterion keys are normalized to sorted descriptions.
OPAQUE = re.compile(r"(?:k_[0-9a-f]{10}|o_(?:[0-9a-f]{7}|[0-9a-f]{10}|[0-9a-f]{64})|opt_[0-9a-f]{5}|c_[0-9a-f]{7})")


def projection(question: dict) -> dict | None:
    """Project a question specification onto its content-addressable routing signature."""
    if not isinstance(question, dict):
        raise ValueError("question object required")
    typ = question.get("type")
    if typ not in ("noul", "score", "choice"):
        raise ValueError("unsupported formatter question type")
    q = {k: question[k] for k in ("type", "instructions", "criteria") if k in question}
    if typ == "noul" and q.get("criteria") in (None, {}):
        q.pop("criteria", None)
    if typ == "choice":
        c = q.get("criteria")
        if not isinstance(c, dict) or not c or not all(isinstance(k, str) for k in c):
            return None
        values = list(c.values())
        if not all(isinstance(v, str) for v in values) or len(set(values)) != len(values):
            return None
        if all(OPAQUE.fullmatch(k) for k in c):
            return {"type": "choice", "instructions": q.get("instructions"), "criteria_descriptions": sorted(values)}
        return {"type": "choice", "instructions": q.get("instructions"), "criteria_named": c}
    return q


def load_registry(path: str | Path) -> dict:
    """Load an alternative question-routing registry JSON."""
    return json.loads(Path(path).read_text(encoding="utf-8"))


def route(question: dict, *, trusted_task: str | None = None, criterion_id: str | None = None) -> str:
    """Route a question definition strictly to ``W0`` or ``W1``."""
    proj = projection(question)
    if proj is None:
        return "W0"
    h = hashlib.sha256(canon(proj).encode("utf-8")).hexdigest()
    entry = R["definitions"].get(h)
    if entry is None or entry["projection"] != proj:
        return "W0"
    if criterion_id is not None and (criterion_id not in IDS or IDS[criterion_id] != entry):
        return "W0"
    if trusted_task is not None and (trusted_task not in R["explicit_tasks"] or trusted_task != entry["task"]):
        return "W0"
    return entry["route"]

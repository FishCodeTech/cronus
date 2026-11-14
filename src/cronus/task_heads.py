"""Public semantic task-head lookup for multi-role adapter routing."""

from __future__ import annotations

import hashlib
import json

from cronus.paths import default_config_path
from cronus.router import canon, projection

SCHEMA = json.loads(default_config_path("head_schema.json").read_text(encoding="utf-8"))


def head_for(question: dict) -> str:
    """Return the active role head name (``decision``, ``scope``, ``permission``, or ``W0``)."""
    p = projection(question)
    if p is None:
        return "W0"
    h = hashlib.sha256(canon(p).encode("utf-8")).hexdigest()
    e = SCHEMA["contracts"].get(h)
    if e is None or e["projection"] != p:
        return "W0"
    return e["head"]

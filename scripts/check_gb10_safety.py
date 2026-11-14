#!/usr/bin/env python3
"""Static AST safety checker for Cronus-9B Python scripts on unified-memory GPUs."""

from __future__ import annotations

import argparse
import ast
import json
from pathlib import Path
import sys

MAX_MEM_FRACTION = 0.35


def _call_name(func: ast.AST) -> str:
    parts: list[str] = []
    cur = func
    while isinstance(cur, ast.Attribute):
        parts.append(cur.attr)
        cur = cur.value
    if isinstance(cur, ast.Name):
        parts.append(cur.id)
    return ".".join(reversed(parts))


def check_python_file(path: Path) -> list[str]:
    """Validate a Python file against unified-memory GPU safety invariants."""
    errors: list[str] = []
    text = path.read_text(encoding="utf-8")
    tree = ast.parse(text, filename=str(path))
    mods: set[str] = set()
    names: set[str] = set()
    attrs: set[str] = set()

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            mods.update(a.name.split(".")[0] for a in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            mods.add(node.module.split(".")[0])
        elif isinstance(node, ast.Name):
            names.add(node.id)
        elif isinstance(node, ast.Attribute):
            attrs.add(node.attr)

    for node in ast.walk(tree):
        if isinstance(node, (ast.For, ast.While)):
            has_empty = any(
                isinstance(sub, ast.Call) and _call_name(sub.func).endswith("empty_cache")
                for sub in ast.walk(node)
            )
            if has_empty:
                errors.append(f"{path}:{node.lineno}: empty_cache() inside loop is forbidden")
            uses_kv = any(
                isinstance(sub, ast.Call)
                and any(kw.arg == "past_key_values" for kw in sub.keywords)
                for sub in ast.walk(node)
            )
            has_sync = any(
                isinstance(sub, ast.Call) and _call_name(sub.func).endswith("synchronize")
                for sub in ast.walk(node)
            )
            if uses_kv and not has_sync:
                errors.append(f"{path}:{node.lineno}: KV-cached loop must synchronize()")

    if "torch" in mods and ("cuda" in attrs or "load_candidate" in names) and "vllm" not in mods:
        has_frac = any(
            isinstance(n, ast.Call) and _call_name(n.func).endswith("set_per_process_memory_fraction")
            for n in ast.walk(tree)
        )
        if not has_frac:
            errors.append(f"{path}: CUDA script must bound memory via set_per_process_memory_fraction(<= {MAX_MEM_FRACTION})")
    return errors


def main() -> int:
    parser = argparse.ArgumentParser(description="Check Cronus Python files for GB10 GPU safety invariants")
    parser.add_argument("paths", nargs="+", type=Path, help="Python files or directories to scan")
    args = parser.parse_args()

    files: list[Path] = []
    for p in args.paths:
        if p.is_dir():
            files.extend(sorted(p.rglob("*.py")))
        else:
            files.append(p)

    all_errors: list[str] = []
    for f in files:
        all_errors.extend(check_python_file(f))

    print(json.dumps({"checked_files": len(files), "violations": all_errors}, indent=2))
    return 1 if all_errors else 0


if __name__ == "__main__":
    sys.exit(main())

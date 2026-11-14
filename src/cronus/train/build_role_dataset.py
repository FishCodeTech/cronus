#!/usr/bin/env python3
"""Stage-2 lineage-deduplicated dataset pack and role split builder."""

from __future__ import annotations

import argparse
import collections
import hashlib
import json
from pathlib import Path
import string

from transformers import AutoTokenizer

from cronus.runtime import Classifier
from cronus.task_heads import head_for

V2_CRITERIA_MAP = {
    "Permission or necessary context is missing; broad assessment scope is insufficient for this operation.": "unresolved",
    "The standing permission covers this action, or an exact current one-use approval covers it.": "established",
    "An applicable explicit operator prohibition is in force.": "prohibited",
    "Required facts and authorization are established under the policy; the executor must still enforce its checks.": "proceed",
    "Obtain missing permission or material facts before execution.": "ask",
    "An explicit applicable prohibition requires refusing this operation.": "deny",
}


def sha(path: str | Path) -> str:
    """Compute SHA-256 hex digest of a file."""
    with open(path, "rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def canon(value) -> str:
    """Serialize value deterministically for deduplication hashes."""
    return json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"), allow_nan=False)


def semantic_class(info: dict) -> str | None:
    """Infer semantic target class for non-general records."""
    if info.get("pool") == "general":
        raise ValueError("General pool has no semantic-class stratification")
    value = info.get("semantic_target")
    if value is not None:
        return str(value).lower()
    target = info.get("target")
    if target is None:
        return None
    if info["question"]["type"] == "choice":
        mapping = info.get("option_mapping") or {}
        found = [k for k, v in mapping.items() if v == target]
        if len(found) == 1:
            return found[0]
        crit_text = info["question"]["criteria"][target]
        if crit_text in V2_CRITERIA_MAP:
            return V2_CRITERIA_MAP[crit_text]
        raise ValueError("Cannot establish semantic class for " + info["id"])
    return str(target).lower()


def proceed_alias(info: dict) -> str:
    """Resolve the option key corresponding to ``proceed`` in a choice question."""
    mapping = info.get("option_mapping") or {}
    if "proceed" in mapping:
        return mapping["proceed"]
    criteria = info["question"].get("criteria") or {}
    matches = [k for k, v in criteria.items() if V2_CRITERIA_MAP.get(v) == "proceed"]
    if len(matches) != 1:
        raise ValueError("Cannot establish proceed alias for " + info["id"])
    return matches[0]


def main() -> None:
    parser = argparse.ArgumentParser(description="Build leak-free Stage-2 role training/validation packs")
    parser.add_argument("--v40-records", type=Path, required=True)
    parser.add_argument("--v47-records", type=Path, required=True)
    parser.add_argument("--audit-report", type=Path, required=True)
    parser.add_argument("--lineage-lock", type=Path, required=True)
    parser.add_argument("--dev-unlabeled", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path, required=True, help="Path to Qwen/Qwen3.5-9B weights for tokenization")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--v47-pack", type=Path, default=None, help="Optional v47 CPU pack for bit-parity cross-check")
    args = parser.parse_args()

    out_dir = args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    v40 = [json.loads(line) for line in args.v40_records.open(encoding="utf-8") if line.strip()]
    v47 = [json.loads(line) for line in args.v47_records.open(encoding="utf-8") if line.strip()]
    audit = json.loads(args.audit_report.read_text(encoding="utf-8"))
    lin = json.loads(args.lineage_lock.read_text(encoding="utf-8"))

    by_id: dict[str, dict] = {}
    for r in v40:
        by_id[r["id"]] = r
    for r in v47:
        if r["id"] in by_id:
            old = by_id[r["id"]]
            assert old["state"] == r["state"] and old["question"] == r["question"] and old["target"] == r["target"], r["id"]
        by_id[r["id"]] = r

    # Verify zero overlap against development set
    dev_rows = [json.loads(line) for line in args.dev_unlabeled.open(encoding="utf-8") if line.strip()]
    dev_ids = {r["id"] for r in dev_rows}
    dev_sq = {hashlib.sha256(canon({"state": r["state"], "question": r["question"]}).encode("utf-8")).hexdigest() for r in dev_rows}
    assert not (set(by_id) & dev_ids), "ID overlap with DEV!"
    for r in by_id.values():
        sq = hashlib.sha256(canon({"state": r["state"], "question": r["question"]}).encode("utf-8")).hexdigest()
        assert sq not in dev_sq, f"state+question overlap with DEV: {r['id']}"

    conflict_action_ids = {c["action_id"] for c in audit["conflicts"]}
    conflict_worlds = {r["world_id"] for r in v40 if r["id"] in conflict_action_ids}

    parent: dict[str, str] = {}

    def find(x: str) -> str:
        while parent.setdefault(x, x) != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: str, b: str) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    for r in by_id.values():
        wid = "W:" + str(r.get("world_id") or r["id"])
        if r.get("family"):
            union(wid, "F:" + str(r["family"]))
        if r.get("parent_world_id"):
            union(wid, "W:" + str(r["parent_world_id"]))
        if r.get("parent_train_id") and r["parent_train_id"] in by_id:
            pw = "W:" + str(by_id[r["parent_train_id"]].get("world_id") or r["parent_train_id"])
            union(wid, pw)
        if r["id"] in lin["ids"]:
            union(wid, "CV:" + lin["ids"][r["id"]]["cv_group"])

    m = Classifier.__new__(Classifier)
    m.tokenizer = AutoTokenizer.from_pretrained(str(args.model_dir), local_files_only=True, trust_remote_code=False)
    m.native_context = 262144
    m.label_ids = [m.tokenizer.encode(c, add_special_tokens=False)[0] for c in string.ascii_uppercase]

    v47_pack = (
        {x["id"]: x for x in map(json.loads, args.v47_pack.open(encoding="utf-8"))}
        if args.v47_pack is not None and args.v47_pack.exists()
        else {}
    )

    excluded = []
    selected: dict[str, list[dict]] = {"decision": [], "scope": []}
    packed_rows: dict[str, dict] = {}
    token_lengths: dict[str, list[int]] = collections.defaultdict(list)

    for r in sorted(by_id.values(), key=lambda x: x["id"]):
        h = head_for(r["question"])
        if h not in ("decision", "scope"):
            continue
        if r.get("target") is None or r.get("mask", False):
            continue
        if h == "decision" and (r["id"] in conflict_action_ids or r.get("world_id") in conflict_worlds):
            excluded.append(
                {
                    "id": r["id"],
                    "world_id": r.get("world_id"),
                    "family": r.get("family"),
                    "reason": "unknown_proceed_conflict",
                }
            )
            continue

        x = m.pack(r["state"], r["question"])
        assert len(x["ids"]) <= m.native_context
        target = r["target"]
        t_str = str(target).lower() if isinstance(target, bool) else str(target)
        idx = x["names"].index(t_str)
        if r["id"] in v47_pack:
            vp = v47_pack[r["id"]]
            assert x["ids"] == vp["ids"] and x["input_ids_sha256"] == vp["input_sha256"] and idx == vp["target_index"], r["id"]

        sc = semantic_class(r)
        root_comp = find("W:" + str(r.get("world_id") or r["id"]))
        p_idx = None
        if h == "decision":
            pa = proceed_alias(r)
            p_idx = x["names"].index(pa)
            assert sc in ("proceed", "ask", "deny"), (r["id"], sc)
        else:
            assert sc in ("true", "false") and x["names"] == ["true", "false"], (r["id"], sc, x["names"])

        packed_rows[r["id"]] = {
            "id": r["id"],
            "head": h,
            "world_id": r.get("world_id"),
            "family": r.get("family"),
            "root_group": root_comp,
            "suite": r.get("suite"),
            "is_guardtask": r["id"].startswith("guardtask_"),
            "semantic_class": sc,
            "proceed_index": p_idx,
            "ids": x["ids"],
            "input_sha256": x["input_ids_sha256"],
            "tokens": len(x["ids"]),
            "names": x["names"],
            "groups": x["groups"],
            "nativeT": x["temperature"],
            "target_index": idx,
            "target_distribution": [float(j == idx) for j in range(len(x["names"]))],
        }
        selected[h].append(packed_rows[r["id"]])
        token_lengths[h].append(len(x["ids"]))

    pack_out = out_dir / "pack_v52.jsonl"
    with pack_out.open("w", encoding="utf-8") as f:
        for rid in sorted(packed_rows):
            f.write(json.dumps(packed_rows[rid], separators=(",", ":")) + "\n")

    splits = {}
    for role in ("decision", "scope"):
        items = selected[role]
        by_comp: dict[str, list[dict]] = collections.defaultdict(list)
        for it in items:
            by_comp[it["root_group"]].append(it)
        strata: dict[tuple[bool, str], list[str]] = collections.defaultdict(list)
        for rg, grp_items in by_comp.items():
            has_gt = any(x["is_guardtask"] for x in grp_items)
            sz = "L" if len(grp_items) >= 20 else ("M" if len(grp_items) >= 6 else "S")
            strata[(has_gt, sz)].append(rg)

        val_rgs: set[str] = set()
        for _, rgs in sorted(strata.items()):
            ordered = sorted(rgs, key=lambda g: hashlib.sha256(f"cronus-split\0{role}\0{g}".encode("utf-8")).hexdigest())
            n_val = max(1, round(len(ordered) * 0.20)) if len(ordered) >= 3 else (1 if len(ordered) >= 2 else 0)
            val_rgs.update(ordered[:n_val])

        train_ids = sorted([x["id"] for x in items if x["root_group"] not in val_rgs])
        val_ids = sorted([x["id"] for x in items if x["root_group"] in val_rgs])
        assert not (set(train_ids) & set(val_ids))
        train_fams = {packed_rows[i]["family"] for i in train_ids}
        val_fams = {packed_rows[i]["family"] for i in val_ids}
        assert not (train_fams & val_fams), f"{role} family leak: {train_fams & val_fams}"
        train_worlds = {packed_rows[i]["world_id"] for i in train_ids}
        val_worlds = {packed_rows[i]["world_id"] for i in val_ids}
        assert not (train_worlds & val_worlds), f"{role} world leak"

        rg_names = {}
        for rg, grp_items in by_comp.items():
            fams = sorted({str(x["family"]) for x in grp_items})
            rg_names[rg] = fams[0] if len(fams) == 1 else f"{fams[0]}+{len(fams)-1}"

        splits[role] = {
            "all_n": len(items),
            "train_n": len(train_ids),
            "validation_n": len(val_ids),
            "train": train_ids,
            "validation": val_ids,
            "root_groups": {x["id"]: rg_names[x["root_group"]] for x in items},
            "families": {x["id"]: x["family"] for x in items},
            "semantic_class": {x["id"]: x["semantic_class"] for x in items},
            "proceed_index": {x["id"]: x["proceed_index"] for x in items if role == "decision"},
            "train_class_counts": dict(collections.Counter(packed_rows[i]["semantic_class"] for i in train_ids)),
            "val_class_counts": dict(collections.Counter(packed_rows[i]["semantic_class"] for i in val_ids)),
            "train_families_n": len(train_fams),
            "val_families_n": len(val_fams),
            "val_guardtask_n": sum(1 for i in val_ids if packed_rows[i]["is_guardtask"]),
            "train_guardtask_n": sum(1 for i in train_ids if packed_rows[i]["is_guardtask"]),
        }

    (out_dir / "role_splits_v52.json").write_text(json.dumps(splits, indent=2) + "\n", encoding="utf-8")
    (out_dir / "excluded_conflicts.json").write_text(json.dumps({"count": len(excluded), "rows": excluded}, indent=2) + "\n", encoding="utf-8")

    manifest = {
        "status": "STAGE2_CPU_PACKED_CLEAN_LINEAGE_SPLITS",
        "dev_overlap_ids": 0,
        "dev_overlap_state_question": 0,
        "excluded_conflicts_n": len(excluded),
        "pack_sha256": sha(pack_out),
        "role_splits_sha256": sha(out_dir / "role_splits_v52.json"),
        "v40_sha256": sha(args.v40_records),
        "v47_sha256": sha(args.v47_records),
        "max_tokens": {r: max(v) for r, v in token_lengths.items()},
        "roles": {
            r: {
                k: v
                for k, v in splits[r].items()
                if k not in ("train", "validation", "root_groups", "families", "semantic_class", "proceed_index")
            }
            for r in ("decision", "scope")
        },
    }
    (out_dir / "dataset_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()

"""Export exact rank-concatenated BF16 single-adapter LoRAs and static merged weights.

Builds single-adapter LoRAs on top of `Qwen/Qwen3.5-9B`:
- `default_r8`:   zero-padded uniform r=8, lora_alpha=16 (scale=2.0)
- `decision_r24`: exact rank-concatenation of `default` (r=8) + `role_decision` (r=16)
- `scope_r16`:    exact rank-concatenation of `default` (r=8) + `role_scope` (r=8)

Optionally (`--export-merged-dir`), merges those adapters into static pre-merged BF16
HF checkpoints using PEFT `merge_and_unload()` for runtime-PEFT-free evaluation or
serving with `serve_native.py`.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil

from peft import PeftModel
from safetensors.torch import load_file, save_file
import torch
from transformers import AutoConfig, Qwen3_5ForCausalLM

from cronus.paths import default_config_path

ALL_12_TARGETS = [
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
    "in_proj_qkv",
    "in_proj_z",
    "in_proj_b",
    "in_proj_a",
    "out_proj",
]


def sha256_file(path: Path) -> str:
    """Return SHA-256 hex digest of a file."""
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def make_adapter_config(r: int, lora_alpha: int, target_modules: list[str], base_model_name: str) -> dict:
    """Create a PEFT LORA adapter_config.json mapping."""
    return {
        "alpha_pattern": {},
        "auto_mapping": None,
        "base_model_name_or_path": base_model_name,
        "bias": "none",
        "fan_in_fan_out": False,
        "inference_mode": True,
        "init_lora_weights": True,
        "layer_replication": None,
        "layers_pattern": None,
        "layers_to_transform": None,
        "loftq_config": {},
        "lora_alpha": lora_alpha,
        "lora_bias": False,
        "lora_dropout": 0.0,
        "modules_to_save": None,
        "peft_type": "LORA",
        "r": r,
        "rank_pattern": {},
        "revision": None,
        "target_modules": target_modules,
        "task_type": "CAUSAL_LM",
        "use_dora": False,
        "use_rslora": False,
    }


def build_combined_role(
    def_sd: dict[str, torch.Tensor],
    role_sd: dict[str, torch.Tensor],
    r_def: int,
    r_role: int,
) -> dict[str, torch.Tensor]:
    """Concatenate `(default, role)` along rank dimension with exact scale parity."""
    r_total = r_def + r_role
    combined: dict[str, torch.Tensor] = {}
    a_keys = sorted(k for k in role_sd if k.endswith(".lora_A.weight"))
    for a_key in a_keys:
        b_key = a_key[: -len(".lora_A.weight")] + ".lora_B.weight"
        a_role = role_sd[a_key].to(torch.float32).contiguous()
        b_role = role_sd[b_key].to(torch.float32).contiguous()
        in_dim = a_role.shape[1]
        out_dim = b_role.shape[0]
        if a_key in def_sd:
            a_def = def_sd[a_key].to(torch.float32).contiguous()
            b_def = def_sd[b_key].to(torch.float32).contiguous()
        else:
            a_def = torch.zeros((r_def, in_dim), dtype=torch.float32)
            b_def = torch.zeros((out_dim, r_def), dtype=torch.float32)
        a_cat = torch.cat([a_def, a_role], dim=0).to(torch.bfloat16).contiguous()
        b_cat = torch.cat([b_def, b_role], dim=1).to(torch.bfloat16).contiguous()
        if a_cat.shape != (r_total, in_dim) or b_cat.shape != (out_dim, r_total):
            raise ValueError(f"Unexpected rank shape for {a_key}: {a_cat.shape} / {b_cat.shape}")
        combined[a_key] = a_cat
        combined[b_key] = b_cat
    return combined


def build_uniform_default(
    def_sd: dict[str, torch.Tensor],
    ref_all_sd: dict[str, torch.Tensor],
    r_def: int = 8,
) -> dict[str, torch.Tensor]:
    """Zero-pad missing attention/MLP projections in the default adapter to uniform rank."""
    out: dict[str, torch.Tensor] = {}
    default_7_targets = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
    for a_key in sorted(k for k in ref_all_sd if k.endswith(".lora_A.weight")):
        proj_name = a_key.split(".")[-3]
        if proj_name not in default_7_targets:
            continue
        b_key = a_key[: -len(".lora_A.weight")] + ".lora_B.weight"
        in_dim = ref_all_sd[a_key].shape[1]
        out_dim = ref_all_sd[b_key].shape[0]
        if a_key in def_sd:
            out[a_key] = def_sd[a_key].to(torch.bfloat16).contiguous()
            out[b_key] = def_sd[b_key].to(torch.bfloat16).contiguous()
        else:
            out[a_key] = torch.zeros((r_def, in_dim), dtype=torch.bfloat16)
            out[b_key] = torch.zeros((out_dim, r_def), dtype=torch.bfloat16)
    return out


def write_adapter(out_dir: Path, state_dict: dict[str, torch.Tensor], cfg: dict) -> dict:
    """Persist adapter_model.safetensors and adapter_config.json."""
    out_dir.mkdir(parents=True, exist_ok=True)
    st_path = out_dir / "adapter_model.safetensors"
    cfg_path = out_dir / "adapter_config.json"
    save_file(state_dict, str(st_path))
    cfg_path.write_text(json.dumps(cfg, indent=2, sort_keys=True) + "\n")
    return {
        "dir": str(out_dir),
        "tensor_count": len(state_dict),
        "r": cfg["r"],
        "lora_alpha": cfg["lora_alpha"],
        "weights_sha256": sha256_file(st_path),
        "config_sha256": sha256_file(cfg_path),
    }


def merge_static_checkpoint(base_model_dir: Path, adapter_dir: Path, out_dir: Path) -> None:
    """Materialize a static pre-merged BF16 HF checkpoint using PEFT merge_and_unload on CPU."""
    out_dir.mkdir(parents=True, exist_ok=True)
    cfg = AutoConfig.from_pretrained(str(base_model_dir), local_files_only=True, trust_remote_code=False)
    base = Qwen3_5ForCausalLM.from_pretrained(
        str(base_model_dir),
        config=cfg.get_text_config(),
        torch_dtype=torch.bfloat16,
        device_map="cpu",
        local_files_only=True,
        trust_remote_code=False,
    )
    peft_model = PeftModel.from_pretrained(base, str(adapter_dir), is_trainable=False)
    merged = peft_model.merge_and_unload()
    merged.save_pretrained(str(out_dir), safe_serialization=True)
    for fn in ("tokenizer.json", "tokenizer_config.json", "vocab.json", "merges.txt", "special_tokens_map.json", "chat_template.jinja"):
        src_f = base_model_dir / fn
        if src_f.is_file():
            shutil.copy2(src_f, out_dir / fn)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=default_config_path("cronus_manifest.json"))
    parser.add_argument("--default-dir", type=Path, required=True)
    parser.add_argument("--adapter-base-dir", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--base-model-name", default="Qwen/Qwen3.5-9B")
    parser.add_argument("--base-model-dir", type=Path, default=None, help="Optional BF16 base model path when --export-merged-dir is set.")
    parser.add_argument("--export-merged-dir", type=Path, default=None, help="Optional directory for pre-merged static checkpoints (default, decision, scope).")
    parser.add_argument("--skip-checksums", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    spec = json.loads(args.manifest.read_text())
    def_weights = args.default_dir / "adapter_model.safetensors"
    if not args.skip_checksums and spec.get("source_default_sha256"):
        if sha256_file(def_weights) != spec["source_default_sha256"]:
            raise RuntimeError("Stage-1 default adapter SHA-256 mismatch")

    def resolve_role(rel_or_abs: str) -> Path:
        p = Path(rel_or_abs)
        return p if p.is_absolute() else (args.adapter_base_dir / p)

    dec_dir = resolve_role(spec["adapters"]["decision"]["path"])
    scp_dir = resolve_role(spec["adapters"]["scope"]["path"])

    def load_role_weights(d: Path) -> dict[str, torch.Tensor]:
        p = d / "guard.safetensors"
        if not p.is_file():
            p = d / "adapter_model.safetensors"
        return load_file(str(p))

    def_sd = load_file(str(def_weights))
    dec_sd = load_role_weights(dec_dir)
    scp_sd = load_role_weights(scp_dir)

    default_uniform = build_uniform_default(def_sd, dec_sd, r_def=8)
    decision_comb = build_combined_role(def_sd, dec_sd, r_def=8, r_role=16)
    scope_comb = build_combined_role(def_sd, scp_sd, r_def=8, r_role=8)

    adapters_meta = {
        "default_r8": write_adapter(
            args.out_dir / "default_r8",
            default_uniform,
            make_adapter_config(8, 16, ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"], args.base_model_name),
        ),
        "decision_r24": write_adapter(
            args.out_dir / "decision_r24",
            decision_comb,
            make_adapter_config(24, 48, ALL_12_TARGETS, args.base_model_name),
        ),
        "scope_r16": write_adapter(
            args.out_dir / "scope_r16",
            scope_comb,
            make_adapter_config(16, 32, ALL_12_TARGETS, args.base_model_name),
        ),
    }

    merged_paths: dict[str, str] = {}
    if args.export_merged_dir is not None:
        if args.base_model_dir is None:
            raise ValueError("--base-model-dir is required when --export-merged-dir is set")
        for role_name, adapter_key in (("default", "default_r8"), ("decision", "decision_r24"), ("scope", "scope_r16")):
            target = args.export_merged_dir / role_name
            merge_static_checkpoint(args.base_model_dir, Path(adapters_meta[adapter_key]["dir"]), target)
            merged_paths[role_name] = str(target)

    receipt = {
        "status": "ok",
        "adapters": adapters_meta,
        "merged_static_checkpoints": merged_paths,
    }
    args.out_dir.mkdir(parents=True, exist_ok=True)
    rec_path = args.out_dir / "combined_lora_receipt.json"
    rec_path.write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps({"status": "ok", "receipt_path": str(rec_path), "receipt": receipt}, indent=2))


if __name__ == "__main__":
    main()

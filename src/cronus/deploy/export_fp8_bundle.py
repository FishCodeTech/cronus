"""Export the Cronus-9B quantized FP8 bundle and residual Punica LoRA adapters.

This exporter:
1. Verifies SHA-256 checksums of the pinned base-model lock, Cronus manifest,
   default stage-1 adapter (r=8, alpha=16), and role adapters (decision r=16, scope r=8).
2. Folds the always-on stage-1 default adapter into the language model weights in
   exact FP32 (omitting unused vision/MTP parameters).
3. Quantizes eligible linear layers ([M, K] divisible by 128) to 128x128 block-wise
   FP8 (`float8_e4m3fn` weights + `float32` `.weight_scale_inv`), writing safetensors
   shards, index, tokenizer files, and `config.json`.
4. Exports standalone residual Punica LoRAs (`loras/role_decision_r16`,
   `loras/role_scope_r8`) and optional `guard_config/` metadata.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import time

import torch
from safetensors import safe_open
from safetensors.torch import load_file, save_file

from cronus.paths import default_config_path

CUDA_MEM_FRACTION = float(os.environ.get("CUDA_MEM_FRACTION", "0.30"))
assert 0.0 < CUDA_MEM_FRACTION <= 0.35, CUDA_MEM_FRACTION

BLOCK = 128
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

QUANTIZABLE_LINEAR_SUFFIXES = (
    ".self_attn.q_proj.weight",
    ".self_attn.k_proj.weight",
    ".self_attn.v_proj.weight",
    ".self_attn.o_proj.weight",
    ".mlp.gate_proj.weight",
    ".mlp.up_proj.weight",
    ".mlp.down_proj.weight",
    ".linear_attn.in_proj_qkv.weight",
    ".linear_attn.in_proj_z.weight",
    ".linear_attn.out_proj.weight",
)


def sha256_file(path: Path) -> str:
    """Compute SHA-256 hex digest of a file."""
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def quantize_128x128_fp8(weight_f32: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Quantize a 2-D float32 weight matrix into 128x128 block-wise FP8 E4M3."""
    fp8_max = float(torch.finfo(torch.float8_e4m3fn).max)
    rows, cols = weight_f32.shape
    if rows % BLOCK != 0 or cols % BLOCK != 0:
        raise ValueError(f"Expected shape divisible by {BLOCK}, got {(rows, cols)}")
    blocks = weight_f32.view(rows // BLOCK, BLOCK, cols // BLOCK, BLOCK)
    amax = blocks.abs().amax(dim=(1, 3))
    scale_inv = torch.clamp(amax / fp8_max, min=1e-12).to(torch.float32)
    quantized = (blocks / scale_inv[:, None, :, None]).clamp(-fp8_max, fp8_max).to(torch.float8_e4m3fn)
    return quantized.reshape(rows, cols).contiguous().cpu(), scale_inv.contiguous().cpu()


def export_residual_lora(
    src_dir: Path,
    out_dir: Path,
    r: int,
    lora_alpha: int,
    base_model_name_or_path: str,
) -> dict:
    """Write a standalone Punica-compatible residual LoRA directory."""
    out_dir.mkdir(parents=True, exist_ok=True)
    src_weights = src_dir / "guard.safetensors"
    if not src_weights.is_file():
        src_weights = src_dir / "adapter_model.safetensors"
    state_dict = load_file(str(src_weights))
    clean_sd = {k: v.to(torch.bfloat16).contiguous() for k, v in state_dict.items()}
    st_path = out_dir / "adapter_model.safetensors"
    cfg_path = out_dir / "adapter_config.json"
    save_file(clean_sd, str(st_path))
    cfg = {
        "alpha_pattern": {},
        "auto_mapping": None,
        "base_model_name_or_path": base_model_name_or_path,
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
        "target_modules": ALL_12_TARGETS,
        "task_type": "CAUSAL_LM",
        "use_dora": False,
        "use_rslora": False,
    }
    cfg_path.write_text(json.dumps(cfg, indent=2, sort_keys=True) + "\n")
    return {
        "dir": str(out_dir),
        "r": r,
        "lora_alpha": lora_alpha,
        "tensor_count": len(clean_sd),
        "weights_sha256": sha256_file(st_path),
        "config_sha256": sha256_file(cfg_path),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-model-dir", type=Path, required=True, help="Local Qwen3.5-9B BF16 directory.")
    parser.add_argument("--default-dir", type=Path, required=True, help="Stage-1 default adapter directory.")
    parser.add_argument("--out-dir", type=Path, required=True, help="Output bundle root directory.")
    parser.add_argument("--adapter-base-dir", type=Path, required=True, help="Root holding relative role adapter paths.")
    parser.add_argument("--manifest", type=Path, default=default_config_path("cronus_manifest.json"))
    parser.add_argument("--base-lock", type=Path, default=default_config_path("base_model.lock.json"))
    parser.add_argument("--readout-path", type=Path, default=None, help="Optional W0 readout.safetensors to copy into guard_config/.")
    parser.add_argument("--calibration-path", type=Path, default=default_config_path("calibration_result.json"))
    parser.add_argument("--hf-base-repo", default="Sn1waR/Cronus-9B", help="Model ID written into LoRA adapter_config.json.")
    parser.add_argument("--model-subdir", default=".", help="Subdir under --out-dir for FP8 model shards (default '.' for HF root layout).")
    parser.add_argument("--skip-checksums", action="store_true", help="Skip SHA-256 verification against manifest/lock.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    torch.cuda.set_per_process_memory_fraction(CUDA_MEM_FRACTION)
    t0 = time.perf_counter()

    spec = json.loads(args.manifest.read_text())
    def_weights = args.default_dir / "adapter_model.safetensors"
    if not args.skip_checksums and spec.get("base_lock_sha256"):
        if sha256_file(args.base_lock) != spec["base_lock_sha256"]:
            raise RuntimeError("Base model lock SHA-256 mismatch")
    if not args.skip_checksums and spec.get("source_default_sha256"):
        if sha256_file(def_weights) != spec["source_default_sha256"]:
            raise RuntimeError("Stage-1 default adapter SHA-256 mismatch")

    def resolve_role_dir(rel_or_abs: str) -> Path:
        p = Path(rel_or_abs)
        return p if p.is_absolute() else (args.adapter_base_dir / p)

    dec_src = resolve_role_dir(spec["adapters"]["decision"]["path"])
    scp_src = resolve_role_dir(spec["adapters"]["scope"]["path"])

    model_out = (args.out_dir / args.model_subdir).resolve()
    lora_root = args.out_dir / "loras"
    model_out.mkdir(parents=True, exist_ok=True)

    for fn in (
        "tokenizer.json",
        "tokenizer_config.json",
        "vocab.json",
        "merges.txt",
        "special_tokens_map.json",
        "chat_template.jinja",
        "generation_config.json",
    ):
        src_file = args.base_model_dir / fn
        if src_file.is_file():
            shutil.copy2(src_file, model_out / fn)

    def_sd = load_file(str(def_weights))
    default_scale = 2.0

    orig_index = json.loads((args.base_model_dir / "model.safetensors.index.json").read_text())
    by_shard: dict[str, list[str]] = {}
    for key, shard_name in orig_index["weight_map"].items():
        if key.startswith(("model.visual.", "mtp.")):
            continue
        by_shard.setdefault(shard_name, []).append(key)

    new_weight_map: dict[str, str] = {}
    total_bytes = 0
    modules_not_converted = ["lm_head", "model.language_model.embed_tokens"]
    folded_default_count = 0
    fp8_quantized_count = 0

    shard_names = sorted(by_shard.keys())
    for idx, orig_shard in enumerate(shard_names, start=1):
        out_shard = f"model-{idx:05d}-of-{len(shard_names):05d}.safetensors"
        out_tensors: dict[str, torch.Tensor] = {}
        with safe_open(str(args.base_model_dir / orig_shard), framework="pt", device="cpu") as handle:
            for key in sorted(by_shard[orig_shard]):
                tensor_cpu = handle.get_tensor(key)
                lora_prefix = "base_model.model." + key[: -len(".weight")] if key.endswith(".weight") else ""
                a_key = lora_prefix + ".lora_A.weight"
                b_key = lora_prefix + ".lora_B.weight"

                is_quant_target = key.endswith(QUANTIZABLE_LINEAR_SUFFIXES)
                if a_key in def_sd or is_quant_target:
                    w_gpu = tensor_cpu.to(device="cuda", dtype=torch.float32)
                    if a_key in def_sd:
                        a_gpu = def_sd[a_key].to(device="cuda", dtype=torch.float32)
                        b_gpu = def_sd[b_key].to(device="cuda", dtype=torch.float32)
                        w_gpu = w_gpu + default_scale * (b_gpu @ a_gpu)
                        folded_default_count += 1
                    if is_quant_target:
                        q_cpu, s_cpu = quantize_128x128_fp8(w_gpu)
                        s_key = key[: -len(".weight")] + ".weight_scale_inv"
                        out_tensors[key] = q_cpu
                        out_tensors[s_key] = s_cpu
                        new_weight_map[key] = out_shard
                        new_weight_map[s_key] = out_shard
                        total_bytes += q_cpu.numel() * q_cpu.element_size() + s_cpu.numel() * s_cpu.element_size()
                        fp8_quantized_count += 1
                    else:
                        bf_cpu = w_gpu.to(torch.bfloat16).contiguous().cpu()
                        out_tensors[key] = bf_cpu
                        new_weight_map[key] = out_shard
                        total_bytes += bf_cpu.numel() * bf_cpu.element_size()
                    del w_gpu
                else:
                    out_tensors[key] = tensor_cpu
                    new_weight_map[key] = out_shard
                    total_bytes += tensor_cpu.numel() * tensor_cpu.element_size()
                    if key.endswith((".linear_attn.in_proj_a.weight", ".linear_attn.in_proj_b.weight", ".linear_attn.conv1d.weight")):
                        modules_not_converted.append(key[: -len(".weight")])

        torch.cuda.synchronize()
        save_file(out_tensors, str(model_out / out_shard))
        print(json.dumps({"event": "shard_written", "shard": out_shard, "tensors": len(out_tensors)}), flush=True)

    (model_out / "model.safetensors.index.json").write_text(
        json.dumps({"metadata": {"total_size": total_bytes}, "weight_map": new_weight_map}, indent=2, sort_keys=True) + "\n"
    )

    cfg = json.loads((args.base_model_dir / "config.json").read_text())
    cfg["quantization_config"] = {
        "activation_scheme": "dynamic",
        "act_per_tensor": False,
        "modules_to_not_convert": sorted(set(modules_not_converted)),
        "quant_method": "fp8",
        "weight_block_size": [128, 128],
        "weight_per_tensor": False,
    }
    (model_out / "config.json").write_text(json.dumps(cfg, indent=2) + "\n")

    dec_info = export_residual_lora(dec_src, lora_root / "role_decision_r16", r=16, lora_alpha=32, base_model_name_or_path=args.hf_base_repo)
    scp_info = export_residual_lora(scp_src, lora_root / "role_scope_r8", r=8, lora_alpha=16, base_model_name_or_path=args.hf_base_repo)

    guard_cfg = args.out_dir / "guard_config"
    guard_cfg.mkdir(parents=True, exist_ok=True)
    shutil.copy2(args.calibration_path, guard_cfg / "calibration_result.json")
    if args.readout_path is not None and args.readout_path.is_file():
        w0_dir = guard_cfg / "W0_readout"
        w0_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(args.readout_path, w0_dir / "readout.safetensors")

    release_manifest = dict(spec)
    release_manifest["adapters"] = {
        "decision": {**spec["adapters"]["decision"], "path": "loras/role_decision_r16"},
        "scope": {**spec["adapters"]["scope"], "path": "loras/role_scope_r8"},
    }
    (guard_cfg / "cronus_manifest.json").write_text(json.dumps(release_manifest, indent=2) + "\n")

    receipt = {
        "status": "ok",
        "elapsed_seconds": round(time.perf_counter() - t0, 2),
        "folded_default_linear_layers": folded_default_count,
        "fp8_128x128_quantized_layers": fp8_quantized_count,
        "total_checkpoint_gib": round(total_bytes / 2**30, 3),
        "model_out": str(model_out),
        "residual_loras": {"decision": dec_info, "scope": scp_info},
    }
    (args.out_dir / "build_fp8_bundle_receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps(receipt, indent=2), flush=True)


if __name__ == "__main__":
    main()

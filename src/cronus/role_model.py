"""Multi-role LoRA candidate loader and exact adapter router for Stage-2 validation."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

from peft import LoraConfig, get_peft_model_state_dict, set_peft_model_state_dict
from safetensors.torch import load_file
import torch

from cronus.paths import default_config_path
from cronus.runtime import Classifier
from cronus.task_heads import head_for

CUDA_MEM_FRACTION = float(os.environ.get("CUDA_MEM_FRACTION", "0.33"))
ALL_PROJS = {
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
}


def sha(path: str | Path) -> str:
    """Compute SHA-256 hex digest of a file."""
    with open(path, "rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def _resolve_relative(path_str: str, base_dir: Path | None) -> Path:
    p = Path(path_str)
    if p.is_absolute() or base_dir is None:
        return p
    return base_dir / p


def load_candidate(
    manifest_path: str | Path | None = None,
    *,
    base_model_dir: str | Path | None = None,
    base_lock: str | Path | None = None,
    source_default_dir: str | Path | None = None,
    readout_path: str | Path | None = None,
    adapter_base_dir: str | Path | None = None,
    verify_checksums: bool = True,
):
    """Load Qwen3.5-9B + Stage-1 ``default`` + Stage-2 role guards with tensor-level verification."""
    if not (0.0 < CUDA_MEM_FRACTION <= 0.35):
        raise RuntimeError(f"CUDA_MEM_FRACTION={CUDA_MEM_FRACTION} exceeds GB10 safety ceiling 0.35")
    if torch.cuda.is_available():
        torch.cuda.set_per_process_memory_fraction(CUDA_MEM_FRACTION)

    mpath = Path(manifest_path) if manifest_path is not None else default_config_path("cronus_manifest.json")
    spec = json.loads(mpath.read_text(encoding="utf-8"))

    model_dir = (
        Path(base_model_dir)
        if base_model_dir is not None
        else Path(os.environ.get("CRONUS_BASE_MODEL_DIR", "Qwen/Qwen3.5-9B"))
    )
    lock_file = Path(base_lock) if base_lock is not None else default_config_path("base_model.lock.json")
    source = (
        Path(source_default_dir)
        if source_default_dir is not None
        else Path(os.environ["CRONUS_SOURCE_DEFAULT_DIR"])
    )
    wpath = (
        Path(readout_path)
        if readout_path is not None
        else Path(os.environ["CRONUS_W0_READOUT_PATH"])
    )
    role_root = Path(adapter_base_dir) if adapter_base_dir is not None else mpath.parent

    if verify_checksums:
        assert sha(source / "adapter_model.safetensors") == spec["source_default_sha256"]
        assert sha(wpath) == spec["W0_readout_sha256"]

    m = Classifier(model_dir, lock_file, adapter=source, training=False)
    w = load_file(str(wpath))
    assert set(w) == {"W0", "label_ids"} and torch.equal(
        m.W.cpu().contiguous().view(torch.uint8), w["W0"].contiguous().view(torch.uint8)
    )
    assert m.label_ids == w["label_ids"].tolist() and m.bias is None

    source_state = load_file(str(source / "adapter_model.safetensors"))
    actual = get_peft_model_state_dict(m.model, adapter_name="default")
    assert set(actual) == set(source_state) and len(actual) == 176
    for k in actual:
        assert torch.equal(
            actual[k].detach().cpu().contiguous().view(torch.uint8), source_state[k].contiguous().view(torch.uint8)
        ), k

    m.role_guard_names = {}
    proof = {"default_tensors_exact": 176, "guards": {}, "original_W0_exact": True}
    assert set(spec["adapters"]) <= {"decision", "scope"}
    for role, e in spec["adapters"].items():
        cdir = _resolve_relative(e["path"], role_root)
        weights_file = cdir / ("guard.safetensors" if (cdir / "guard.safetensors").exists() else "adapter_model.safetensors")
        if verify_checksums and "guard_sha256" in e:
            assert sha(weights_file) == e["guard_sha256"]
        if verify_checksums and "config_sha256" in e:
            assert sha(cdir / "adapter_config.json") == e["config_sha256"]
        cfg = LoraConfig.from_pretrained(str(cdir))
        tm = cfg.target_modules
        assert cfg.r in (8, 16) and cfg.lora_alpha == 2 * cfg.r and cfg.bias == "none"
        assert tm == "all-linear" or set(tm) == ALL_PROJS, tm
        name = "role_" + role
        m.model.add_adapter(name, cfg, autocast_adapter_dtype=True)
        expected = load_file(str(weights_file))
        set_peft_model_state_dict(m.model, expected, adapter_name=name)
        actual_g = get_peft_model_state_dict(m.model, adapter_name=name)
        assert set(actual_g) == set(expected) and len(actual_g) == 496, len(actual_g)
        for k in actual_g:
            assert torch.equal(
                actual_g[k].detach().cpu().contiguous().view(torch.uint8), expected[k].contiguous().view(torch.uint8)
            ), (role, k)
        m.role_guard_names[role] = name
        proof["guards"][role] = {"tensors_exact": 496, "guard_sha256": sha(weights_file)}

    actual = get_peft_model_state_dict(m.model, adapter_name="default")
    assert set(actual) == set(source_state)
    for k in actual:
        assert torch.equal(
            actual[k].detach().cpu().contiguous().view(torch.uint8), source_state[k].contiguous().view(torch.uint8)
        ), k

    m.model.requires_grad_(False)
    m.model.eval()
    activate(m, None)
    return m, spec, proof


def activate(m: Classifier, question: dict | None) -> tuple[str, list[str]]:
    """Switch active PEFT adapters strictly for the detected task head."""
    head = "W0" if question is None else head_for(question)
    names = ["default"] + ([m.role_guard_names[head]] if head in m.role_guard_names else [])
    m.model.base_model.set_adapter(names, inference_mode=True)
    m.model.eval()
    assert all(not p.requires_grad for p in m.model.parameters())
    for layer in m.model.modules():
        if hasattr(layer, "lora_A") and len(layer.lora_A):
            assert list(layer.active_adapters) == names and not layer.disable_adapters
    return head, names


def logits(m: Classifier, state, question: dict):
    """Evaluate a short-context example with automatic task-head adapter activation."""
    x = m.pack(state, question)
    head, names = activate(m, question)
    h = m.hidden_state(x)
    z = m.logits_from_hidden(h, x)
    return z, x, head, names

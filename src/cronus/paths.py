"""Path resolution utilities for Cronus-9B configs, weights, and adapters."""

from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path

PACKAGE_ROOT = Path(__file__).resolve().parent
CONFIGS_DIR = PACKAGE_ROOT / "configs"


def default_config_path(filename: str) -> Path:
    """Return the bundled config file path inside the ``cronus.configs`` package."""
    return CONFIGS_DIR / filename


def _env_path(*names: str) -> Path | None:
    for name in names:
        raw = os.environ.get(name, "").strip()
        if raw:
            return Path(raw)
    return None


@dataclass(frozen=True)
class BundleLayout:
    """Resolved model, LoRA adapter, manifest, and calibration paths."""

    bundle_dir: Path | None
    model_dir: Path
    lora_decision_dir: Path
    lora_scope_dir: Path
    manifest_path: Path
    calibration_path: Path
    w0_readout_path: Path | None


def resolve_bundle_layout(bundle_dir: str | Path | None = None) -> BundleLayout:
    """Resolve model weights and configs for either HF hub layout or local bundle layout.

    Supports:
    1. Hugging Face release layout (`Sn1waR/Cronus-9B`):
       - `<root>/config.json` (model weights at bundle root)
       - `<root>/loras/{role_decision_r16,role_scope_r8}`
       - `<root>/guard_config/{calibration_result.json,cronus_manifest.json,W0_readout/readout.safetensors}`
    2. Local FP8 staging layout:
       - `<root>/model_default_fp8_128x128/config.json`
       - `<root>/{role_decision_r16,role_scope_r8}`
    """
    bdir = Path(bundle_dir) if bundle_dir is not None else _env_path("CRONUS_BUNDLE_DIR", "GUARD_BUNDLE_DIR")

    if bdir is not None and (bdir / "model_default_fp8_128x128" / "config.json").exists():
        default_model = bdir / "model_default_fp8_128x128"
        default_dec = bdir / "role_decision_r16"
        default_scp = bdir / "role_scope_r8"
        default_guard_cfg = bdir / "guard_config"
    elif bdir is not None:
        default_model = bdir
        default_dec = bdir / "loras" / "role_decision_r16"
        default_scp = bdir / "loras" / "role_scope_r8"
        default_guard_cfg = bdir / "guard_config"
    else:
        default_model = Path("Cronus-9B")
        default_dec = default_model / "loras" / "role_decision_r16"
        default_scp = default_model / "loras" / "role_scope_r8"
        default_guard_cfg = CONFIGS_DIR

    model_dir = _env_path("CRONUS_MODEL_DIR", "GUARD_MODEL_DIR") or default_model
    lora_dec = _env_path("CRONUS_LORA_DECISION_DIR", "GUARD_LORA_DECISION_DIR") or default_dec
    lora_scp = _env_path("CRONUS_LORA_SCOPE_DIR", "GUARD_LORA_SCOPE_DIR") or default_scp

    manifest_override = _env_path("CRONUS_MANIFEST_PATH", "GUARD_MANIFEST_PATH")
    if manifest_override is not None:
        manifest_path = manifest_override
    elif (default_guard_cfg / "cronus_manifest.json").exists():
        manifest_path = default_guard_cfg / "cronus_manifest.json"
    else:
        manifest_path = default_config_path("cronus_manifest.json")

    cal_override = _env_path("CRONUS_CALIBRATION_PATH", "GUARD_CALIBRATION_PATH")
    if cal_override is not None:
        calibration_path = cal_override
    elif (default_guard_cfg / "calibration_result.json").exists():
        calibration_path = default_guard_cfg / "calibration_result.json"
    else:
        calibration_path = default_config_path("calibration_result.json")

    w0_override = _env_path("CRONUS_W0_READOUT_PATH", "GUARD_W0_READOUT_PATH")
    if w0_override is not None:
        w0_path: Path | None = w0_override
    elif (default_guard_cfg / "W0_readout" / "readout.safetensors").exists():
        w0_path = default_guard_cfg / "W0_readout" / "readout.safetensors"
    elif (model_dir / "readout.safetensors").exists():
        w0_path = model_dir / "readout.safetensors"
    else:
        w0_path = None

    return BundleLayout(
        bundle_dir=bdir,
        model_dir=model_dir,
        lora_decision_dir=lora_dec,
        lora_scope_dir=lora_scp,
        manifest_path=manifest_path,
        calibration_path=calibration_path,
        w0_readout_path=w0_path,
    )

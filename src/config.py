"""Shared configuration and device selection for the football pipeline."""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "configs" / "football_tft.json"
DATA_CONFIG_KEYS = ("history_schema", "features", "preprocessing", "context_length", "train_size")


def _merge_config(base: dict, overrides: dict) -> dict:
    merged = dict(base)
    for key, value in overrides.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _merge_config(merged[key], value)
        else:
            merged[key] = value
    return merged


def load_config(path: Path) -> dict:
    defaults = json.loads(DEFAULT_CONFIG.read_text(encoding="utf-8"))
    legacy_defaults = defaults.pop("legacy_defaults", {})
    config = json.loads(path.read_text(encoding="utf-8"))
    config.pop("legacy_defaults", None)
    return _merge_config(_merge_config(defaults, legacy_defaults), config)


def data_config_snapshot(config: dict) -> dict:
    return {key: config[key] for key in DATA_CONFIG_KEYS}


def resolve_device(name: str):
    import torch

    if name == "auto":
        name = "cuda" if torch.cuda.is_available() else (
            "mps" if torch.backends.mps.is_available() else "cpu"
        )
    return torch.device(name)

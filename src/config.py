"""Shared configuration and device selection for the football pipeline."""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "configs" / "football_tft.json"

# Defaults for configurations and checkpoints created before these options existed.
NEW_OPTION_DEFAULTS = {
    "feature_path": "data/a_a_a_feature.csv",
    "results_path": "data/a_a_a_test.csv",
    "context_length": 10,
    "distribution": "negative_binomial",
    "gru_layers": 1,
    "gru_dropout": 0.0,
    "optimizer_betas": [0.9, 0.999],
    "optimizer_eps": 1e-8,
    "scheduler_factor": 0.5,
    "scheduler_patience": 2,
    "scheduler_threshold": 0.0001,
    "scheduler_cooldown": 0,
    "scheduler_min_lr": 0.0,
    "inference_quantiles": [0.1, 0.5, 0.9],
    "handicap_output": "artifacts/football_tft_poisson/handicap_predictions.csv",
}


def load_config(path: Path) -> dict:
    config = json.loads(path.read_text(encoding="utf-8"))
    return {**NEW_OPTION_DEFAULTS, **config}


def resolve_device(name: str):
    import torch

    if name == "auto":
        name = "cuda" if torch.cuda.is_available() else (
            "mps" if torch.backends.mps.is_available() else "cpu"
        )
    return torch.device(name)

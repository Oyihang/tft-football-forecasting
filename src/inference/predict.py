from __future__ import annotations

import argparse
import json
import sys
from decimal import Decimal
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.stats import nbinom, poisson

ROOT = Path(__file__).resolve().parents[2]
if __package__ in (None, ""):
    sys.path.insert(0, str(ROOT))

from src.config import DEFAULT_CONFIG, load_config, resolve_device
from src.models.football_tft import build_football_tft
from src.training.trainer import make_loader, predict_params


def quantile_columns(values):
    if not isinstance(values, list) or not values:
        raise ValueError("inference_quantiles must be a non-empty list.")
    columns = []
    seen = set()
    for value in values:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError("Each inference quantile must be a number between 0 and 1.")
        q = Decimal(str(value))
        if not q.is_finite() or not 0 < q < 1:
            raise ValueError("Each inference quantile must be a number between 0 and 1.")
        label = "p" + format((q * 100).normalize(), "f").replace(".", "_")
        if label in seen:
            raise ValueError("inference_quantiles must not contain duplicates.")
        seen.add(label)
        columns.append((float(q), label))
    return columns


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    args = parser.parse_args()
    cfg = load_config(args.config)
    quantiles = quantile_columns(cfg["inference_quantiles"])
    data_dir = ROOT / cfg["processed_dir"]
    checkpoint_path = ROOT / cfg["inference_checkpoint"]
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    device = resolve_device(cfg["inference_device"])
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    schema = ckpt["feature_schema"]
    prepared_schema = json.loads((data_dir / "feature_schema.json").read_text(encoding="utf-8"))
    if prepared_schema != schema:
        raise ValueError("Prepared data schema differs from checkpoint; use matching data and checkpoint.")
    model = build_football_tft(schema, ckpt["model_config"]).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    with np.load(data_dir / "split_indices.npz") as split:
        idx = split["test_idx"]
    home = np.load(data_dir / "home_history.npy", mmap_mode="r")[idx]
    away = np.load(data_dir / "away_history.npy", mmap_mode="r")[idx]
    known = np.load(data_dir / "known.npy", mmap_mode="r")[idx]
    static = np.load(data_dir / "static.npy", mmap_mode="r")[idx]
    targets = np.load(data_dir / "targets.npy", mmap_mode="r")[idx]
    ids = np.load(data_dir / "match_ids.npy", mmap_mode="r")[idx]
    loader = make_loader(home, away, known, static, targets=None,
                         batch_size=cfg["inference_batch_size"], shuffle=False,
                         num_workers=cfg["num_workers"])
    params = predict_params(model, loader, device)
    rows = {"match_id": ids, "actual_home_goals": targets[:, 0], "actual_away_goals": targets[:, 1]}
    for side, index in [("home", 0), ("away", 1)]:
        if model.distribution == "poisson":
            mu = params[:, index].astype(float)
            distribution = poisson(mu)
        else:
            mu = params[:, 2 * index].astype(float)
            alpha = params[:, 2 * index + 1].astype(float)
            n = 1.0 / np.maximum(alpha, 1e-8)
            p = 1.0 / (1.0 + alpha * mu)
            distribution = nbinom(n, p)
        rows[f"pred_{side}_xg"] = mu
        median = distribution.ppf(0.5)
        if not np.isfinite(median).all():
            raise ValueError(f"Non-finite {side} score prediction from the model distribution.")
        rows[f"pred_{side}_goals"] = median.astype(np.int64)
        for q, label in quantiles:
            rows[f"{side}_{label}"] = median if q == 0.5 else distribution.ppf(q)
    out = pd.DataFrame(rows)
    output_path = ROOT / cfg["inference_output"]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(output_path, index=False)
    print(f"Saved {len(out):,} test predictions to {output_path.resolve()}")
    print(out.head(10).to_string(index=False))


if __name__ == "__main__":
    main()

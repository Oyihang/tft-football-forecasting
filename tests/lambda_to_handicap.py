"""Convert expected goals to handicap win/draw/loss probabilities."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import poisson, skellam

ROOT = Path(__file__).resolve().parents[1]
if __package__ in (None, ""):
    sys.path.insert(0, str(ROOT))

from src.config import DEFAULT_CONFIG, load_config


def calculate_prob_handicap(lambda_home, lambda_away, handicap):
    """Compare home goals + handicap with away goals under independent Poisson models."""
    home, away, line = np.broadcast_arrays(
        np.asarray(lambda_home, dtype=float),
        np.asarray(lambda_away, dtype=float),
        np.asarray(handicap, dtype=float),
    )
    if not np.isfinite(home).all() or not np.isfinite(away).all() or (home < 0).any() or (away < 0).any():
        raise ValueError("Expected goals must be finite and non-negative.")
    if not np.isfinite(line).all():
        raise ValueError("Handicap values must be finite.")

    win = np.empty(home.shape, dtype=float)
    draw = np.zeros(home.shape, dtype=float)
    loss = np.empty(home.shape, dtype=float)
    integer_line = line == np.floor(line)
    both_positive = (home > 0) & (away > 0)
    home_zero = home == 0
    away_zero = (away == 0) & ~home_zero

    threshold = -line[both_positive]
    win[both_positive] = skellam.sf(np.floor(threshold), home[both_positive], away[both_positive])
    loss[both_positive] = skellam.cdf(np.ceil(threshold) - 1, home[both_positive], away[both_positive])
    mask = both_positive & integer_line
    draw[mask] = skellam.pmf(-line[mask], home[mask], away[mask])

    win[home_zero] = poisson.cdf(np.ceil(line[home_zero]) - 1, away[home_zero])
    loss[home_zero] = poisson.sf(np.floor(line[home_zero]), away[home_zero])
    mask = home_zero & integer_line
    draw[mask] = poisson.pmf(line[mask], away[mask])

    threshold = -line[away_zero]
    win[away_zero] = poisson.sf(np.floor(threshold), home[away_zero])
    loss[away_zero] = poisson.cdf(np.ceil(threshold) - 1, home[away_zero])
    mask = away_zero & integer_line
    draw[mask] = poisson.pmf(-line[mask], home[mask])

    probabilities = np.stack((win, draw, loss), axis=-1)
    if not np.isfinite(probabilities).all() or not np.allclose(probabilities.sum(axis=-1), 1.0, atol=1e-8, rtol=0):
        raise ValueError("Could not calculate valid handicap probabilities for all rows.")
    return win, draw, loss


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--input", type=Path, help="Prediction CSV; defaults to inference_output in config.")
    parser.add_argument("--handicap-data", type=Path, help="CSV with match_id and rq; defaults to results_path in config.")
    parser.add_argument("--output", type=Path, help="Output CSV; defaults to handicap_output in config.")
    args = parser.parse_args()
    cfg = load_config(args.config)
    input_path = args.input if args.input is not None else ROOT / cfg["inference_output"]
    handicap_path = args.handicap_data if args.handicap_data is not None else ROOT / cfg["results_path"]
    output_path = args.output if args.output is not None else ROOT / cfg["handicap_output"]
    if output_path.resolve() in (input_path.resolve(), handicap_path.resolve()):
        raise ValueError("Output must be different from the prediction and handicap input files.")

    predictions = pd.read_csv(input_path)
    handicap_data = pd.read_csv(handicap_path, usecols=["match_id", "rq"], low_memory=False)
    required = {"match_id", "pred_home_xg", "pred_away_xg"}
    missing_columns = required - set(predictions.columns)
    if missing_columns:
        raise ValueError(f"Prediction CSV is missing columns: {sorted(missing_columns)}")
    if "rq" in predictions.columns:
        raise ValueError("Prediction CSV already contains rq; use a prediction file without a handicap column.")
    for name, data in (("predictions", predictions), ("handicap data", handicap_data)):
        if data["match_id"].isna().any() or data["match_id"].duplicated().any():
            raise ValueError(f"{name} must have non-null, unique match_id values.")
    if predictions.empty:
        raise ValueError("Prediction CSV contains no matches.")

    merged = predictions.merge(handicap_data, on="match_id", how="left", validate="one_to_one", indicator=True)
    unmatched = merged["_merge"] != "both"
    if unmatched.any():
        raise ValueError(f"Handicap data missing for match_id values: {merged.loc[unmatched, 'match_id'].head().tolist()}")
    merged = merged.drop(columns="_merge")
    win, draw, loss = calculate_prob_handicap(
        merged["pred_home_xg"], merged["pred_away_xg"], merged["rq"]
    )
    merged["home_prob"], merged["draw_prob"], merged["away_prob"] = win, draw, loss
    merged["predicted_result"] = np.array(["HomeWin", "Draw", "AwayWin"])[
        np.argmax(np.stack((win, draw, loss), axis=-1), axis=-1)
    ]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(output_path, index=False)
    print(f"Saved {len(merged):,} handicap predictions to {output_path.resolve()}")


if __name__ == "__main__":
    main()

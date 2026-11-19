"""Prepare match-level football samples from the supplied 20-row history table."""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[2]
if __package__ in (None, ""):
    sys.path.insert(0, str(ROOT))
from src.config import DEFAULT_CONFIG, load_config
from src.data.splits import training_sample_count
EXCLUDED = {"match_id", "seq_id", "id", "team_id", "team_id_oppo",
            "competition_id", "competition_name", "match_time"}
# Historical team-perspective features. The first 10 rows are home-team history;
# the last 10 rows are away-team history.
OBSERVED_COLUMNS = [
    "competition_type", "is_neutral", "win_rate_last_5", "days_since_last_match",
    "is_double_week", "rank", "rank_oppo", "half_goals", "half_goals_oppo",
    "goals", "goals_oppo", "red_cards", "red_cards_oppo", "yellow_cards",
    "yellow_cards_oppo", "corner_kicks", "corner_kicks_oppo", "venue_capacity",
    "venue_surface", "stage_id", "season_id", "group_count", "round_count",
    "group_num", "round_num", "round_progress", "weather", "pressure",
    "temperature", "wind", "humidity",
]
# Match-level pre-match context from the result table. Betting/odds and live
# columns are intentionally excluded (eu_*, rq*, live*) to avoid leakage and
# to keep the model independent of betting markets.
STATIC_NUMERIC = [
    "rank", "rank_oppo", "venue_capacity", "group_count", "round_count",
    "group_num", "round_num", "round_progress", "pressure", "temperature",
    "wind", "humidity",
]
STATIC_CATEGORICAL = [
    "competition_type", "venue_id", "venue_surface", "stage_id", "season_id",
    "weather",
]
TARGETS = ["home_scores", "away_scores"]

def parse_dt(series):
    # Source timestamps use day/month/year strings.
    return pd.to_datetime(series, dayfirst=True, errors="coerce")

def numeric_frame(df, columns):
    out = pd.DataFrame(index=df.index)
    for c in columns:
        out[c] = pd.to_numeric(df[c], errors="coerce") if c in df else np.nan
    return out.replace([np.inf, -np.inf], np.nan)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    ap.add_argument("--feature", type=Path)
    ap.add_argument("--results", type=Path)
    ap.add_argument("--output-dir", type=Path)
    ap.add_argument("--train-size", type=int)
    args = ap.parse_args()
    cfg = load_config(args.config)
    args.feature = args.feature if args.feature is not None else ROOT / cfg["feature_path"]
    args.results = args.results if args.results is not None else ROOT / cfg["results_path"]
    args.output_dir = args.output_dir if args.output_dir is not None else ROOT / cfg["processed_dir"]
    args.train_size = args.train_size if args.train_size is not None else cfg["train_size"]
    context_length = cfg["context_length"]
    if not isinstance(context_length, int) or isinstance(context_length, bool) or not 1 <= context_length <= 10:
        raise ValueError("context_length must be an integer between 1 and 10.")

    feat = pd.read_csv(args.feature, low_memory=False)
    result = pd.read_csv(args.results, low_memory=False)
    required_f = {"match_id", "seq_id", *OBSERVED_COLUMNS}
    required_r = {"match_id", "match_time", *TARGETS, *STATIC_NUMERIC, *STATIC_CATEGORICAL}
    missing_f, missing_r = required_f-set(feat.columns), required_r-set(result.columns)
    if missing_f or missing_r:
        raise ValueError(f"Missing feature columns={sorted(missing_f)}; result columns={sorted(missing_r)}")
    if result.match_id.duplicated().any():
        raise ValueError("Result table contains duplicate match_id values.")

    # Strict one-to-one match_id alignment and exactly 20 history rows per match.
    counts = feat.groupby("match_id").size()
    bad = counts[counts != 20]
    if len(bad):
        raise ValueError(f"Expected exactly 20 history rows per match; invalid examples: {bad.head().to_dict()}")
    if set(feat.match_id) != set(result.match_id):
        raise ValueError("Feature/result match_id sets do not match exactly.")

    result = result.copy()
    result["_target_dt"] = parse_dt(result["match_time"])
    if result["_target_dt"].isna().any():
        raise ValueError("Some result match_time values could not be parsed.")
    result = result.sort_values(["_target_dt", "match_id"]).reset_index(drop=True)
    if args.train_size <= 0 or args.train_size >= len(result):
        raise ValueError(f"train-size must be between 1 and {len(result)-1}; got {args.train_size}")
    fit_size = training_sample_count(args.train_size, cfg["validation_fraction"])
    # Order each team's own 10 matches from oldest to newest. Source seq_id is
    # newest-to-oldest within each block, so reverse each block independently.
    feat = feat.sort_values(["match_id", "seq_id"]).copy()
    feat["_hist_dt"] = parse_dt(feat["match_time"])
    if feat["_hist_dt"].isna().any():
        raise ValueError("Some history match_time values could not be parsed.")
    joined = feat.merge(result[["match_id", "_target_dt"]], on="match_id", validate="many_to_one")
    if (joined["_hist_dt"] >= joined["_target_dt"]).any():
        raise ValueError("History contains a match at/after the target match; possible temporal leakage.")
    if (feat.groupby("match_id")["seq_id"].nunique() != 20).any():
        raise ValueError("Each match must contain seq_id 1..20 exactly once.")
    seq_stats = feat.groupby("match_id")["seq_id"].agg(["min", "max", "nunique"])
    if not ((seq_stats["min"] == 1) & (seq_stats["max"] == 20) & (seq_stats["nunique"] == 20)).all():
        raise ValueError("seq_id must be exactly 1..20 for every match.")

    # Check team orientation against the result table: rows 1..10 are home,
    # rows 11..20 are away. IDs are validation keys only, never model inputs.
    orient = feat.groupby("match_id").agg(first_team=("team_id","first"), last_team=("team_id","last"))
    result_index = result.set_index("match_id")
    if not (orient.first_team == result_index.loc[orient.index, "team_id"]).all():
        raise ValueError("seq_id 1..10 do not consistently represent the result-table home team.")
    if not (orient.last_team == result_index.loc[orient.index, "team_id_oppo"]).all():
        raise ValueError("seq_id 11..20 do not consistently represent the result-table away team.")

    # Reindex once into result-table chronological order, then reshape.
    ordered_index = pd.MultiIndex.from_product([result.match_id.to_list(), range(1, 21)],
                                                names=["match_id", "seq_id"])
    ordered_feat = feat.set_index(["match_id", "seq_id"]).reindex(ordered_index).reset_index()
    numeric = numeric_frame(ordered_feat, OBSERVED_COLUMNS).to_numpy(np.float32)
    numeric = numeric.reshape(len(result), 20, len(OBSERVED_COLUMNS))
    xh = numeric[:, :context_length, :]
    xa = numeric[:, 10:10 + context_length, :]
    # Source seq_id is newest-to-oldest; reverse each team block separately.
    xh = xh[:, ::-1, :].copy()
    xa = xa[:, ::-1, :].copy()

    # Fit all preprocessing on the gradient-training subset, not validation or test.
    all_hist = np.concatenate([xh, xa], axis=0)
    hist_train_rows = np.concatenate([xh[:fit_size], xa[:fit_size]], axis=0).reshape(-1, len(OBSERVED_COLUMNS))
    hist_all_rows = all_hist.reshape(-1, len(OBSERVED_COLUMNS))
    med = np.nanmedian(hist_train_rows, axis=0)
    med = np.where(np.isfinite(med), med, 0.0)
    hist_train_filled = np.where(np.isfinite(hist_train_rows), hist_train_rows, med)
    hist_all_filled = np.where(np.isfinite(hist_all_rows), hist_all_rows, med)
    scaler_hist = StandardScaler().fit(hist_train_filled)
    hist_scaled = scaler_hist.transform(hist_all_filled).astype(np.float32).reshape(2*len(result), context_length, len(OBSERVED_COLUMNS))
    xh_scaled, xa_scaled = hist_scaled[:len(result)], hist_scaled[len(result):]

    # One known time feature: normalized history position. It is deterministic
    # and available for all selected encoder steps and the single decoder step.
    positions = np.linspace(0.0, 1.0, context_length + 1, dtype=np.float32)
    known = np.tile(positions[None, :, None], (len(result), 1, 1))

    # Static continuous covariates.
    static_raw = numeric_frame(result, STATIC_NUMERIC)
    static_med = static_raw.iloc[:fit_size].median().fillna(0.0)
    static_filled = static_raw.fillna(static_med)
    scaler_static = StandardScaler().fit(static_filled.iloc[:fit_size].to_numpy(np.float32))
    static_cont = scaler_static.transform(static_filled.to_numpy(np.float32)).astype(np.float32)

    # Static categorical embeddings: unknown/missing=0; train categories start at 1.
    cat_maps = {}
    cat_arrays = []
    for col in STATIC_CATEGORICAL:
        train_values = result.iloc[:fit_size][col].fillna("__MISSING__").astype(str)
        categories = sorted(train_values.unique().tolist())
        mapping = {v:i+1 for i,v in enumerate(categories)}
        cat_maps[col] = mapping
        values = result[col].fillna("__MISSING__").astype(str).map(mapping).fillna(0).to_numpy(np.int64)
        cat_arrays.append(values)
    static_cat = np.stack(cat_arrays, axis=1).astype(np.float32)
    static = np.concatenate([static_cont, static_cat], axis=1).astype(np.float32)

    y = result[TARGETS].to_numpy(np.float32)
    if not np.isfinite(y).all() or (y < 0).any() or (y != np.floor(y)).any():
        raise ValueError("Goal counts must be finite non-negative integers.")
    # Training pool = first train_size matches; test = all later matches.
    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)
    np.save(out/"home_history.npy", xh_scaled)
    np.save(out/"away_history.npy", xa_scaled)
    np.save(out/"known.npy", known)
    np.save(out/"static.npy", static)
    np.save(out/"targets.npy", y)
    np.save(out/"match_ids.npy", result.match_id.to_numpy())
    np.save(out/"match_times.npy", result["_target_dt"].astype("int64").to_numpy())
    np.savez(out/"split_indices.npz", train_pool_idx=np.arange(args.train_size),
             test_idx=np.arange(args.train_size, len(result)))
    schema = {
        "observed_history_columns": OBSERVED_COLUMNS,
        "known_inputs": ["history_position"],
        "static_numeric_columns": STATIC_NUMERIC,
        "static_categorical_columns": STATIC_CATEGORICAL,
        "static_columns_ordered": STATIC_NUMERIC + STATIC_CATEGORICAL,
        "category_maps": cat_maps,
        "category_cardinalities": [len(cat_maps[c])+1 for c in STATIC_CATEGORICAL],
        "targets": TARGETS, "context_length": context_length, "horizon_length": 1,
        "train_size": args.train_size, "preprocessing_train_size": fit_size, "total_matches": len(result),
        "split_strategy": "chronological: first train_size matches train pool, remainder held-out test",
        "excluded_columns": sorted(EXCLUDED | {"eu_home","eu_draw","eu_away","rq","rq_home","rq_away","live","live_over","live_under"}),
        "scalers": {
            "history_median": med.tolist(), "history_mean": scaler_hist.mean_.tolist(),
            "history_scale": scaler_hist.scale_.tolist(),
            "static_median": static_med.to_dict(), "static_mean": scaler_static.mean_.tolist(),
            "static_scale": scaler_static.scale_.tolist()
        }
    }
    (out/"feature_schema.json").write_text(json.dumps(schema,ensure_ascii=False,indent=2),encoding="utf-8")
    print(f"Prepared {len(result):,} matches | train pool={args.train_size:,} | preprocessing fit={fit_size:,} | test={len(result)-args.train_size:,}")
    print(f"home={xh_scaled.shape}, away={xa_scaled.shape}, known={known.shape}, static={static.shape}, y={y.shape}")
    print(f"History features ({len(OBSERVED_COLUMNS)}): {OBSERVED_COLUMNS}")
    print(f"Static numeric ({len(STATIC_NUMERIC)}): {STATIC_NUMERIC}")
    print(f"Static categorical ({len(STATIC_CATEGORICAL)}): {STATIC_CATEGORICAL}")
    print("Betting odds and live fields are excluded.")

if __name__ == "__main__":
    main()

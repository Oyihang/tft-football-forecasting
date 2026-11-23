"""Prepare match-level football samples from the configured history table."""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[2]
if __package__ in (None, ""):
    sys.path.insert(0, str(ROOT))
from src.config import DEFAULT_CONFIG, data_config_snapshot, load_config
from src.data.splits import training_sample_count


def parse_dt(series, day_first):
    return pd.to_datetime(series, dayfirst=day_first, errors="coerce")

def numeric_frame(df, columns):
    out = pd.DataFrame(index=df.index)
    for c in columns:
        out[c] = pd.to_numeric(df[c], errors="coerce") if c in df else np.nan
    return out.replace([np.inf, -np.inf], np.nan)


def categorical_frame(df, columns, category_maps, unknown_index):
    encoded = np.full((len(df), len(columns)), unknown_index, dtype=np.int64)
    for index, column in enumerate(columns):
        numeric = pd.to_numeric(df[column], errors="coerce")
        valid = numeric.notna() & np.isfinite(numeric) & (numeric == np.floor(numeric))
        keys = numeric.loc[valid].astype(np.int64).astype(str)
        encoded[valid.to_numpy(), index] = (
            keys.map(category_maps[column]).fillna(unknown_index).to_numpy(dtype=np.int64)
        )
    return encoded


def scale_array(values, median, scaler):
    shape = values.shape
    rows = values.reshape(-1, shape[-1])
    filled = np.where(np.isfinite(rows), rows, median)
    return scaler.transform(filled).astype(np.float32).reshape(shape)


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
    history_cfg = cfg["history_schema"]
    feature_cfg = cfg["features"]
    preprocessing_cfg = cfg["preprocessing"]
    observed_numeric_columns = feature_cfg["observed"]["numeric"]
    observed_categorical_columns = feature_cfg["observed"]["categorical"]
    observed_columns = observed_numeric_columns + observed_categorical_columns
    known_numeric_columns = feature_cfg["known"]["numeric"]
    known_categorical_columns = feature_cfg["known"]["categorical"]
    away_view_swaps = feature_cfg["known"]["away_view_swaps"]
    position_feature = feature_cfg["known"]["position"]
    known_history_columns = known_numeric_columns + known_categorical_columns
    static_numeric_columns = feature_cfg["static"]["numeric"]
    static_categorical_columns = feature_cfg["static"]["categorical"]
    category_columns = feature_cfg["category_order"]
    category_values = feature_cfg["category_values"]
    targets = feature_cfg["targets"]
    excluded_columns = feature_cfg["excluded_columns"]
    match_id_column = history_cfg["match_id_column"]
    sequence_id_column = history_cfg["sequence_id_column"]
    feature_time_column = history_cfg["feature_time_column"]
    result_time_column = history_cfg["result_time_column"]
    home_team_id_column = history_cfg["home_team_id_column"]
    away_team_id_column = history_cfg["away_team_id_column"]
    horizon_length = history_cfg["horizon_length"]
    rows_per_team = history_cfg["history_rows_per_team"]
    home_sequence_start = history_cfg["home_sequence_start"]
    away_sequence_start = history_cfg["away_sequence_start"]
    unknown_category_index = preprocessing_cfg["unknown_category_index"]
    category_index_start = preprocessing_cfg["category_index_start"]
    empty_numeric_median = preprocessing_cfg["empty_numeric_median"]
    position_start, position_end = preprocessing_cfg["history_position_range"]
    if not isinstance(rows_per_team, int) or isinstance(rows_per_team, bool) or rows_per_team < 1:
        raise ValueError("history_rows_per_team must be a positive integer.")
    if not isinstance(home_sequence_start, int) or not isinstance(away_sequence_start, int):
        raise ValueError("Home and away sequence starts must be integers.")
    if horizon_length != 1:
        raise ValueError("This football model currently supports a horizon_length of 1.")
    if away_sequence_start != home_sequence_start + rows_per_team:
        raise ValueError("Away history must immediately follow the home history in seq_id order.")
    if (not isinstance(context_length, int) or isinstance(context_length, bool)
            or not 1 <= context_length <= rows_per_team):
        raise ValueError(
            f"context_length must be an integer between 1 and {rows_per_team}."
        )
    if unknown_category_index != 0 or category_index_start <= unknown_category_index:
        raise ValueError("Category index 0 is reserved for unknown; known categories must start above it.")
    if set(category_columns) != set(category_values):
        raise ValueError("category_order and category_values must define the same category fields.")
    used_category_columns = set(
        observed_categorical_columns + known_categorical_columns + static_categorical_columns
    )
    if not used_category_columns.issubset(category_values):
        raise ValueError("Every categorical input must have configured category_values.")
    for column, values in category_values.items():
        if not values or len(values) != len(set(values)):
            raise ValueError(f"category_values[{column!r}] must contain unique values.")

    feat = pd.read_csv(args.feature, low_memory=False)
    result = pd.read_csv(args.results, low_memory=False)
    required_f = {
        match_id_column, sequence_id_column, feature_time_column, home_team_id_column,
        *observed_columns, *known_history_columns,
    }
    required_r = {
        match_id_column, result_time_column, *targets, *static_numeric_columns,
        *static_categorical_columns, *known_numeric_columns, *known_categorical_columns,
        home_team_id_column, away_team_id_column,
    }
    missing_f, missing_r = required_f-set(feat.columns), required_r-set(result.columns)
    if missing_f or missing_r:
        raise ValueError(f"Missing feature columns={sorted(missing_f)}; result columns={sorted(missing_r)}")
    if result[match_id_column].duplicated().any():
        raise ValueError("Result table contains duplicate match_id values.")

    expected_history_rows = rows_per_team * 2
    expected_home_seq_ids = list(range(home_sequence_start, home_sequence_start + rows_per_team))
    expected_away_seq_ids = list(range(away_sequence_start, away_sequence_start + rows_per_team))
    expected_seq_ids = expected_home_seq_ids + expected_away_seq_ids
    # Strict one-to-one match alignment and configured history row counts.
    counts = feat.groupby(match_id_column).size()
    bad = counts[counts != expected_history_rows]
    if len(bad):
        raise ValueError(
            f"Expected exactly {expected_history_rows} history rows per match; "
            f"invalid examples: {bad.head().to_dict()}"
        )
    if set(feat[match_id_column]) != set(result[match_id_column]):
        raise ValueError("Feature/result match_id sets do not match exactly.")

    result = result.copy()
    result["_target_dt"] = parse_dt(
        result[result_time_column], history_cfg["timestamps_day_first"]
    )
    if result["_target_dt"].isna().any():
        raise ValueError("Some result match_time values could not be parsed.")
    result = result.sort_values(["_target_dt", match_id_column]).reset_index(drop=True)
    if args.train_size <= 0 or args.train_size >= len(result):
        raise ValueError(f"train-size must be between 1 and {len(result)-1}; got {args.train_size}")
    fit_size = training_sample_count(args.train_size, cfg["validation_fraction"])
    # Source sequence IDs are newest-to-oldest within each team block.
    feat = feat.sort_values([match_id_column, sequence_id_column]).copy()
    feat["_hist_dt"] = parse_dt(
        feat[feature_time_column], history_cfg["timestamps_day_first"]
    )
    if feat["_hist_dt"].isna().any():
        raise ValueError("Some history match_time values could not be parsed.")
    joined = feat.merge(
        result[[match_id_column, "_target_dt"]],
        on=match_id_column,
        validate="many_to_one",
    )
    if (joined["_hist_dt"] >= joined["_target_dt"]).any():
        raise ValueError("History contains a match at/after the target match; possible temporal leakage.")
    if (feat.groupby(match_id_column)[sequence_id_column].nunique() != expected_history_rows).any():
        raise ValueError("Each match must contain every configured seq_id exactly once.")
    seq_stats = feat.groupby(match_id_column)[sequence_id_column].agg(["min", "max", "nunique"])
    if not (
        (seq_stats["min"] == expected_seq_ids[0])
        & (seq_stats["max"] == expected_seq_ids[-1])
        & (seq_stats["nunique"] == expected_history_rows)
    ).all():
        raise ValueError("Each match must contain the configured contiguous seq_id range.")

    # Check configured home/away history blocks against the result table.
    orient = feat.groupby(match_id_column).agg(
        first_team=(home_team_id_column, "first"),
        last_team=(home_team_id_column, "last"),
    )
    result_index = result.set_index(match_id_column)
    if not (orient.first_team == result_index.loc[orient.index, home_team_id_column]).all():
        raise ValueError("Configured home history does not match the result-table home team.")
    if not (orient.last_team == result_index.loc[orient.index, away_team_id_column]).all():
        raise ValueError("Configured away history does not match the result-table away team.")

    # Reindex once into result-table chronological order, then reshape.
    ordered_index = pd.MultiIndex.from_product(
        [result[match_id_column].to_list(), expected_seq_ids],
        names=[match_id_column, sequence_id_column],
    )
    ordered_feat = (
        feat.set_index([match_id_column, sequence_id_column])
        .reindex(ordered_index)
        .reset_index()
    )
    numeric_columns = observed_numeric_columns + known_numeric_columns
    numeric = numeric_frame(ordered_feat, numeric_columns).to_numpy(np.float32)
    numeric = numeric.reshape(len(result), expected_history_rows, len(numeric_columns))
    observed_numeric = numeric[:, :, :len(observed_numeric_columns)]
    known_numeric = numeric[:, :, len(observed_numeric_columns):]
    category_maps = {
        column: {
            str(value): index + category_index_start
            for index, value in enumerate(category_values[column])
        }
        for column in category_columns
    }
    categorical = categorical_frame(
        ordered_feat, category_columns, category_maps, unknown_category_index
    )
    categorical = categorical.reshape(len(result), expected_history_rows, len(category_columns))

    xh = observed_numeric[:, :context_length, :]
    away_start = rows_per_team
    xa = observed_numeric[:, away_start:away_start + context_length, :]
    home_known = known_numeric[:, :context_length, :]
    away_known = known_numeric[:, away_start:away_start + context_length, :]
    observed_category_indices = [
        category_columns.index(column) for column in observed_categorical_columns
    ]
    xh_cat = categorical[:, :context_length, observed_category_indices]
    xa_cat = categorical[:, away_start:away_start + context_length, observed_category_indices]
    known_category_indices = [category_columns.index(column) for column in known_categorical_columns]
    home_known_categories = categorical[:, :context_length, known_category_indices]
    away_known_categories = categorical[
        :, away_start:away_start + context_length, known_category_indices
    ]
    # Source seq_id is newest-to-oldest; reverse each team block separately.
    xh = xh[:, ::-1, :].copy()
    xa = xa[:, ::-1, :].copy()
    home_known = home_known[:, ::-1, :].copy()
    away_known = away_known[:, ::-1, :].copy()
    xh_cat = xh_cat[:, ::-1, :].copy()
    xa_cat = xa_cat[:, ::-1, :].copy()
    home_known_categories = home_known_categories[:, ::-1, :].copy()
    away_known_categories = away_known_categories[:, ::-1, :].copy()

    # Fit all preprocessing on the gradient-training subset, not validation or test.
    observed_train_rows = np.concatenate([xh[:fit_size], xa[:fit_size]], axis=0).reshape(
        -1, len(observed_numeric_columns)
    )
    observed_median = np.nanmedian(observed_train_rows, axis=0)
    observed_median = np.where(
        np.isfinite(observed_median), observed_median, empty_numeric_median
    )
    observed_scaler = StandardScaler().fit(
        np.where(np.isfinite(observed_train_rows), observed_train_rows, observed_median)
    )
    xh_scaled = scale_array(xh, observed_median, observed_scaler)
    xa_scaled = scale_array(xa, observed_median, observed_scaler)
    xh_scaled = np.concatenate([xh_scaled, xh_cat.astype(np.float32)], axis=-1)
    xa_scaled = np.concatenate([xa_scaled, xa_cat.astype(np.float32)], axis=-1)

    # Known values from each team's history remain separate; target-match values
    # fill the final decoder step in home/away perspective order.
    target_known = numeric_frame(result, known_numeric_columns).to_numpy(np.float32)
    away_target_known = target_known.copy()
    for home_feature, away_feature in away_view_swaps:
        home_index = known_numeric_columns.index(home_feature)
        away_index = known_numeric_columns.index(away_feature)
        away_target_known[:, home_index] = target_known[:, away_index]
        away_target_known[:, away_index] = target_known[:, home_index]
    known_fit_rows = np.concatenate([
        home_known[:fit_size].reshape(-1, len(known_numeric_columns)),
        away_known[:fit_size].reshape(-1, len(known_numeric_columns)),
        target_known[:fit_size],
        away_target_known[:fit_size],
    ], axis=0)
    known_median = np.nanmedian(known_fit_rows, axis=0)
    known_median = np.where(np.isfinite(known_median), known_median, empty_numeric_median)
    known_scaler = StandardScaler().fit(
        np.where(np.isfinite(known_fit_rows), known_fit_rows, known_median)
    )
    home_known_scaled = scale_array(home_known, known_median, known_scaler)
    away_known_scaled = scale_array(away_known, known_median, known_scaler)
    home_target_scaled = scale_array(target_known[:, None, :], known_median, known_scaler)
    away_target_scaled = scale_array(away_target_known[:, None, :], known_median, known_scaler)
    home_known_full = np.concatenate([home_known_scaled, home_target_scaled], axis=1)
    away_known_full = np.concatenate([away_known_scaled, away_target_scaled], axis=1)

    target_known_categories = categorical_frame(
        result, known_categorical_columns, category_maps, unknown_category_index
    )
    home_known_categories_full = np.concatenate([
        home_known_categories, target_known_categories[:, None, :],
    ], axis=1)
    away_known_categories_full = np.concatenate([
        away_known_categories, target_known_categories[:, None, :],
    ], axis=1)
    positions = np.linspace(
        position_start, position_end, context_length + 1, dtype=np.float32
    )
    history_position = np.tile(positions[None, :, None], (len(result), 1, 1))
    known_numeric_inputs = np.concatenate([
        history_position, home_known_full, away_known_full,
    ], axis=-1)
    known = np.concatenate([
        known_numeric_inputs,
        home_known_categories_full.astype(np.float32),
        away_known_categories_full.astype(np.float32),
    ], axis=-1)

    # Static continuous covariates.
    static_raw = numeric_frame(result, static_numeric_columns)
    static_med = static_raw.iloc[:fit_size].median().fillna(empty_numeric_median)
    static_filled = static_raw.fillna(static_med)
    scaler_static = StandardScaler().fit(static_filled.iloc[:fit_size].to_numpy(np.float32))
    static_cont = scaler_static.transform(static_filled.to_numpy(np.float32)).astype(np.float32)

    static_cat = categorical_frame(
        result, static_categorical_columns, category_maps, unknown_category_index
    ).astype(np.float32)
    static = np.concatenate([static_cont, static_cat], axis=1).astype(np.float32)

    y = result[targets].to_numpy(np.float32)
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
    np.save(out/"match_ids.npy", result[match_id_column].to_numpy())
    np.save(out/"match_times.npy", result["_target_dt"].astype("int64").to_numpy())
    np.savez(out/"split_indices.npz", train_pool_idx=np.arange(args.train_size),
             test_idx=np.arange(args.train_size, len(result)))
    known_numeric_input_columns = (
        [position_feature]
        + [f"home_{column}" for column in known_numeric_columns]
        + [f"away_{column}" for column in known_numeric_columns]
    )
    known_categorical_input_columns = [
        f"{side}_{column}"
        for side in ("home", "away")
        for column in known_categorical_columns
    ]
    category_cardinality = lambda column: max(
        [unknown_category_index, *category_maps[column].values()]
    ) + 1
    schema = {
        "observed_history_columns": observed_columns,
        "observed_numeric_columns": observed_numeric_columns,
        "observed_categorical_columns": observed_categorical_columns,
        "observed_category_cardinalities": [
            category_cardinality(column) for column in observed_categorical_columns
        ],
        "known_history_columns": known_history_columns,
        "known_inputs": known_numeric_input_columns + known_categorical_input_columns,
        "known_numeric_columns": known_numeric_input_columns,
        "known_categorical_columns": known_categorical_input_columns,
        "known_category_cardinalities": [
            category_cardinality(column) for column in known_categorical_columns
        ] * 2,
        "static_numeric_columns": static_numeric_columns,
        "static_categorical_columns": static_categorical_columns,
        "static_columns_ordered": static_numeric_columns + static_categorical_columns,
        "static_category_cardinalities": [
            category_cardinality(column) for column in static_categorical_columns
        ],
        "category_columns": category_columns,
        "category_cardinalities": [category_cardinality(column) for column in category_columns],
        "category_maps": category_maps,
        "targets": targets, "context_length": context_length, "horizon_length": horizon_length,
        "data_config": data_config_snapshot(cfg),
        "train_size": args.train_size, "preprocessing_train_size": fit_size, "total_matches": len(result),
        "split_strategy": "chronological: first train_size matches train pool, remainder held-out test",
        "excluded_columns": sorted(excluded_columns),
        "scalers": {
            "history_median": observed_median.tolist(), "history_mean": observed_scaler.mean_.tolist(),
            "history_scale": observed_scaler.scale_.tolist(),
            "known_median": known_median.tolist(), "known_mean": known_scaler.mean_.tolist(),
            "known_scale": known_scaler.scale_.tolist(),
            "static_median": static_med.to_dict(), "static_mean": scaler_static.mean_.tolist(),
            "static_scale": scaler_static.scale_.tolist()
        }
    }
    (out/"feature_schema.json").write_text(json.dumps(schema,ensure_ascii=False,indent=2),encoding="utf-8")
    print(f"Prepared {len(result):,} matches | train pool={args.train_size:,} | preprocessing fit={fit_size:,} | test={len(result)-args.train_size:,}")
    print(f"home={xh_scaled.shape}, away={xa_scaled.shape}, known={known.shape}, static={static.shape}, y={y.shape}")
    print(f"Observed numeric ({len(observed_numeric_columns)}): {observed_numeric_columns}")
    print(f"Observed categorical ({len(observed_categorical_columns)}): {observed_categorical_columns}")
    print(f"Known input ({len(schema['known_inputs'])}): {schema['known_inputs']}")
    print(f"Static numeric ({len(static_numeric_columns)}): {static_numeric_columns}")
    print(f"Static categorical ({len(static_categorical_columns)}): {static_categorical_columns}")
    print("Betting odds and live fields are excluded.")

if __name__ == "__main__":
    main()

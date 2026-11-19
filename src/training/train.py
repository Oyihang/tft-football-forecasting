from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
if __package__ in (None, ""):
    sys.path.insert(0, str(ROOT))

from src.config import DEFAULT_CONFIG, load_config, resolve_device
from src.data.splits import training_sample_count
from src.models.football_tft import build_football_tft
from src.training.metrics import count_metrics
from src.training.trainer import make_loader, run_epoch


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--skip-test", action="store_true", help="Do not evaluate the held-out test set.")
    args = parser.parse_args()
    cfg = load_config(args.config)
    seed_all(cfg["seed"])

    data_dir = ROOT / cfg["processed_dir"]
    schema = json.loads((data_dir / "feature_schema.json").read_text(encoding="utf-8"))
    if int(schema["train_size"]) != int(cfg["train_size"]):
        raise ValueError("Config train_size differs from prepared data; rerun prepare_football.py.")
    if schema["context_length"] != cfg["context_length"]:
        raise ValueError("Config context_length differs from prepared data; rerun prepare_football.py.")
    home = np.load(data_dir / "home_history.npy")
    away = np.load(data_dir / "away_history.npy")
    known = np.load(data_dir / "known.npy")
    static = np.load(data_dir / "static.npy")
    targets = np.load(data_dir / "targets.npy")
    with np.load(data_dir / "split_indices.npz") as split:
        train_pool = split["train_pool_idx"]
        test_idx = split["test_idx"]

    fit_size = training_sample_count(len(train_pool), cfg["validation_fraction"])
    if schema.get("preprocessing_train_size") != fit_size:
        raise ValueError("Prepared data preprocessing split differs from training split or is outdated; rerun prepare_football.py.")
    train_idx = train_pool[:fit_size]
    val_idx = train_pool[fit_size:]
    if cfg["epochs"] < 1:
        raise ValueError("epochs must be at least 1.")
    device = resolve_device(cfg["device"])
    print(f"Device={device} | optimizer_train={len(train_idx)} | validation={len(val_idx)} | heldout_test={len(test_idx)}")
    print(f"Shapes home={home.shape}, away={away.shape}, known={known.shape}, static={static.shape}, y={targets.shape}")

    def loader(indices, shuffle):
        return make_loader(
            home[indices], away[indices], known[indices], static[indices], targets[indices],
            cfg["batch_size"], shuffle, cfg["num_workers"],
        )

    train_loader = loader(train_idx, True)
    val_loader = loader(val_idx, False)
    model_config = {
        key: cfg[key]
        for key in ("hidden_dim", "num_heads", "lstm_layers", "dropout", "gru_layers", "gru_dropout", "distribution")
    }
    model = build_football_tft(schema, model_config).to(device)
    nll_label = "Poisson" if model.distribution == "poisson" else "NB"
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=cfg["learning_rate"], weight_decay=cfg["weight_decay"],
        betas=tuple(cfg["optimizer_betas"]), eps=cfg["optimizer_eps"],
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=cfg["scheduler_factor"], patience=cfg["scheduler_patience"],
        threshold=cfg["scheduler_threshold"], cooldown=cfg["scheduler_cooldown"],
        min_lr=cfg["scheduler_min_lr"],
    )
    best = float("inf")
    stale = 0
    history = []
    ckpt_dir = ROOT / cfg["checkpoint_dir"]
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    best_path = ckpt_dir / "best_model.pt"
    for epoch in range(1, cfg["epochs"] + 1):
        tr_loss, tr_pred, tr_y = run_epoch(model, train_loader, device, optimizer, cfg["grad_clip"])
        va_loss, va_pred, va_y = run_epoch(model, val_loader, device)
        scheduler.step(va_loss)
        row = {
            "epoch": epoch, "train_nll": tr_loss, "val_nll": va_loss,
            "train_metrics": count_metrics(tr_y, tr_pred), "val_metrics": count_metrics(va_y, va_pred),
            "lr": optimizer.param_groups[0]["lr"],
        }
        history.append(row)
        print(f"Epoch {epoch:03d}/{cfg['epochs']} | train {nll_label}-NLL={tr_loss:.4f} | val {nll_label}-NLL={va_loss:.4f} | "
              f"val home MAE={row['val_metrics']['home_MAE']:.4f} away MAE={row['val_metrics']['away_MAE']:.4f}")
        if va_loss < best:
            best = va_loss
            stale = 0
            torch.save({
                "model_state_dict": model.state_dict(), "model_config": model_config,
                "feature_schema": schema, "config": cfg, "best_epoch": epoch, "val_nll": best,
            }, best_path)
        else:
            stale += 1
            if stale >= cfg["patience"]:
                print(f"Early stopping after {cfg['patience']} epochs without validation improvement.")
                break

    (ckpt_dir / "training_history.json").write_text(
        json.dumps(history, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    if args.skip_test:
        print(f"Test evaluation skipped; any existing test metrics were not updated. Best checkpoint: {best_path}")
        return
    ckpt = torch.load(best_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    test_loss, test_pred, test_y = run_epoch(model, loader(test_idx, False), device)
    metrics = {f"test_{nll_label}_NLL": test_loss, **count_metrics(test_y, test_pred)}
    (ckpt_dir / "test_metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print("Held-out test metrics:", json.dumps(metrics, indent=2))
    print(f"Best checkpoint: {best_path}")


if __name__ == "__main__":
    main()

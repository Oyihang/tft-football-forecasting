import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.stats import poisson

from src.config import DEFAULT_CONFIG, load_config
from src.data.prepare_football import OBSERVED_COLUMNS, STATIC_CATEGORICAL, STATIC_NUMERIC
from src.inference.predict import quantile_columns


class PipelineConfigTest(unittest.TestCase):
    def test_quantiles(self):
        self.assertEqual(quantile_columns([0.1, 0.025]), [(0.1, "p10"), (0.025, "p2_5")])
        for invalid in ([], [0], [1], [0.1, 0.1], ["0.1"]):
            with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                quantile_columns(invalid)

    def test_legacy_config_defaults(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "legacy.json"
            path.write_text('{"train_size": 5}', encoding="utf-8")
            cfg = load_config(path)
        self.assertEqual(cfg["train_size"], 5)
        self.assertEqual(cfg["context_length"], 10)
        self.assertEqual(cfg["gru_layers"], 1)
        self.assertEqual(cfg["distribution"], "negative_binomial")

    def test_configured_pipeline(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            features = []
            results = []
            for match_id in range(4):
                result = {column: 1 for column in STATIC_NUMERIC + STATIC_CATEGORICAL}
                if match_id == 2:
                    result["rank"] = 1000
                    result["competition_type"] = 999
                result.update(match_id=match_id, match_time="10/01/2020",
                              home_scores=1, away_scores=2,
                              team_id=match_id * 2, team_id_oppo=match_id * 2 + 1)
                results.append(result)
                for seq_id in range(1, 21):
                    feature = {column: seq_id for column in OBSERVED_COLUMNS}
                    if match_id == 2:
                        feature = {column: seq_id + 100 for column in OBSERVED_COLUMNS}
                    feature.update(match_id=match_id, seq_id=seq_id, match_time="01/01/2020",
                                   team_id=match_id * 2 + (seq_id > 10))
                    features.append(feature)
            feature_path = root / "features.csv"
            results_path = root / "results.csv"
            pd.DataFrame(features).to_csv(feature_path, index=False)
            pd.DataFrame(results).to_csv(results_path, index=False)

            cfg = load_config(DEFAULT_CONFIG)
            cfg.update(
                feature_path=str(feature_path), results_path=str(results_path),
                processed_dir=str(root / "processed"), checkpoint_dir=str(root / "checkpoints"),
                inference_checkpoint=str(root / "checkpoints" / "best_model.pt"),
                inference_output=str(root / "predictions.csv"), train_size=3, context_length=3,
                validation_fraction=0.34, epochs=1, batch_size=2, inference_batch_size=2,
                hidden_dim=4, num_heads=1, lstm_layers=2, gru_layers=2, gru_dropout=0.1,
                dropout=0.0, device="cpu", inference_device="cpu",
                inference_quantiles=[0.1, 0.025],
            )
            config_path = root / "config.json"
            config_path.write_text(json.dumps(cfg), encoding="utf-8")
            project = DEFAULT_CONFIG.parent.parent
            for module in ("src.data.prepare_football", "src.training.train", "src.inference.predict"):
                result = subprocess.run(
                    [sys.executable, "-X", "utf8", "-s", "-m", module, "--config", str(config_path)],
                    cwd=project, capture_output=True, text=True, encoding="utf-8",
                )
                self.assertEqual(result.returncode, 0, f"{module} failed:\n{result.stdout}\n{result.stderr}")

            schema = json.loads((root / "processed" / "feature_schema.json").read_text(encoding="utf-8"))
            home = np.load(root / "processed" / "home_history.npy")
            away = np.load(root / "processed" / "away_history.npy")
            self.assertEqual(home.shape[:2], (4, 3))
            self.assertEqual(schema["context_length"], 3)
            self.assertEqual(schema["preprocessing_train_size"], 2)
            checkpoint = torch.load(root / "checkpoints" / "best_model.pt", map_location="cpu", weights_only=False)
            self.assertEqual(checkpoint["model_config"]["gru_layers"], 2)
            self.assertEqual(checkpoint["model_config"]["gru_dropout"], 0.1)
            self.assertEqual(checkpoint["model_config"]["distribution"], "poisson")
            self.assertIn("test_Poisson_NLL", json.loads(
                (root / "checkpoints" / "test_metrics.json").read_text(encoding="utf-8")
            ))
            mean = schema["scalers"]["history_mean"][0]
            scale = schema["scalers"]["history_scale"][0]
            self.assertEqual(mean, 7.0)
            self.assertEqual(schema["scalers"]["static_median"]["rank"], 1.0)
            static = np.load(root / "processed" / "static.npy")
            self.assertEqual(static[2, len(STATIC_NUMERIC)], 0)
            np.testing.assert_allclose(home[0, :, 0] * scale + mean, [3, 2, 1], atol=1e-5)
            np.testing.assert_allclose(away[0, :, 0] * scale + mean, [13, 12, 11], atol=1e-5)
            predictions = pd.read_csv(root / "predictions.csv")
            self.assertEqual(
                list(predictions.columns),
                ["match_id", "actual_home_goals", "actual_away_goals",
                 "pred_home_xg", "pred_home_goals", "home_p10", "home_p2_5",
                 "pred_away_xg", "pred_away_goals", "away_p10", "away_p2_5"],
            )
            for side in ("home", "away"):
                rates = predictions[f"pred_{side}_xg"].to_numpy()
                self.assertTrue((rates > 0).all())
                np.testing.assert_array_equal(predictions[f"pred_{side}_goals"], poisson.ppf(0.5, rates))
                np.testing.assert_array_equal(predictions[f"{side}_p10"], poisson.ppf(0.1, rates))
                np.testing.assert_array_equal(predictions[f"{side}_p2_5"], poisson.ppf(0.025, rates))

            nb_cfg = dict(
                cfg, distribution="negative_binomial", checkpoint_dir=str(root / "nb"),
                inference_checkpoint=str(root / "nb" / "best_model.pt"),
                inference_output=str(root / "nb_predictions.csv"),
                inference_quantiles=[0.1, 0.5],
            )
            nb_config_path = root / "nb_config.json"
            nb_config_path.write_text(json.dumps(nb_cfg), encoding="utf-8")
            for module in ("src.training.train", "src.inference.predict"):
                nb_run = subprocess.run(
                    [sys.executable, "-X", "utf8", "-s", "-m", module, "--config", str(nb_config_path)],
                    cwd=project, capture_output=True, text=True, encoding="utf-8",
                )
                self.assertEqual(nb_run.returncode, 0, f"{module} failed:\n{nb_run.stderr}")
            self.assertIn("test_NB_NLL", json.loads(
                (root / "nb" / "test_metrics.json").read_text(encoding="utf-8")
            ))
            nb_predictions = pd.read_csv(root / "nb_predictions.csv")
            self.assertEqual(len(nb_predictions), 1)
            for side in ("home", "away"):
                np.testing.assert_array_equal(nb_predictions[f"pred_{side}_goals"], nb_predictions[f"{side}_p50"])

            schema_path = root / "processed" / "feature_schema.json"
            altered_schema = dict(schema)
            altered_schema["preprocessing_train_size"] = 1
            schema_path.write_text(json.dumps(altered_schema), encoding="utf-8")
            mismatch_inference = subprocess.run(
                [sys.executable, "-X", "utf8", "-s", "-m", "src.inference.predict", "--config", str(config_path)],
                cwd=project, capture_output=True, text=True, encoding="utf-8",
            )
            self.assertNotEqual(mismatch_inference.returncode, 0)
            self.assertIn("schema differs from checkpoint", mismatch_inference.stderr)
            schema_path.write_text(json.dumps(schema), encoding="utf-8")

            skip_cfg = dict(cfg, checkpoint_dir=str(root / "skip_test"))
            skip_path = root / "skip_config.json"
            skip_path.write_text(json.dumps(skip_cfg), encoding="utf-8")
            skipped = subprocess.run(
                [sys.executable, "-X", "utf8", "-s", "-m", "src.training.train",
                 "--config", str(skip_path), "--skip-test"],
                cwd=project, capture_output=True, text=True, encoding="utf-8",
            )
            self.assertEqual(skipped.returncode, 0, skipped.stderr)
            self.assertTrue((root / "skip_test" / "best_model.pt").exists())
            self.assertFalse((root / "skip_test" / "test_metrics.json").exists())

            cfg["validation_fraction"] = 0.7
            config_path.write_text(json.dumps(cfg), encoding="utf-8")
            mismatch_split = subprocess.run(
                [sys.executable, "-X", "utf8", "-s", "-m", "src.training.train", "--config", str(config_path)],
                cwd=project, capture_output=True, text=True, encoding="utf-8",
            )
            self.assertNotEqual(mismatch_split.returncode, 0)
            self.assertIn("preprocessing split differs", mismatch_split.stderr)
            cfg["validation_fraction"] = 0.34
            cfg["context_length"] = 4
            config_path.write_text(json.dumps(cfg), encoding="utf-8")
            mismatch = subprocess.run(
                [sys.executable, "-X", "utf8", "-s", "-m", "src.training.train", "--config", str(config_path)],
                cwd=project, capture_output=True, text=True, encoding="utf-8",
            )
            self.assertNotEqual(mismatch.returncode, 0)
            self.assertIn("context_length differs from prepared data", mismatch.stderr)
            override_dir = root / "overridden"
            overridden = subprocess.run(
                [sys.executable, "-X", "utf8", "-s", "-m", "src.data.prepare_football", "--config", str(config_path),
                 "--train-size", "2", "--output-dir", str(override_dir)],
                cwd=project, capture_output=True, text=True, encoding="utf-8",
            )
            self.assertEqual(overridden.returncode, 0, overridden.stderr)
            override_schema = json.loads((override_dir / "feature_schema.json").read_text(encoding="utf-8"))
            self.assertEqual(override_schema["train_size"], 2)
            self.assertEqual(override_schema["context_length"], 4)

            bad_features = pd.DataFrame(features)
            bad_features.loc[0, "match_time"] = "invalid"
            bad_features.to_csv(feature_path, index=False)
            invalid_date = subprocess.run(
                [sys.executable, "-X", "utf8", "-s", "-m", "src.data.prepare_football", "--config", str(config_path)],
                cwd=project, capture_output=True, text=True, encoding="utf-8",
            )
            self.assertNotEqual(invalid_date.returncode, 0)
            self.assertIn("history match_time values could not be parsed", invalid_date.stderr)
            pd.DataFrame(features).to_csv(feature_path, index=False)

            bad_results = pd.DataFrame(results)
            bad_results["home_scores"] = bad_results["home_scores"].astype(float)
            bad_results.loc[0, "home_scores"] = 1.5
            bad_results.to_csv(results_path, index=False)
            invalid_target = subprocess.run(
                [sys.executable, "-X", "utf8", "-s", "-m", "src.data.prepare_football", "--config", str(config_path)],
                cwd=project, capture_output=True, text=True, encoding="utf-8",
            )
            self.assertNotEqual(invalid_target.returncode, 0)
            self.assertIn("Goal counts must be finite non-negative integers", invalid_target.stderr)


if __name__ == "__main__":
    unittest.main()

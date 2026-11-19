import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import poisson

from lambda_to_handicap import calculate_prob_handicap


class HandicapTest(unittest.TestCase):
    def test_probabilities_match_poisson_sum(self):
        goals = np.arange(60)
        for home, away in [(1.65, 1.43), (0, 0), (0, 1.2), (1.2, 0)]:
            matrix = poisson.pmf(goals, home)[:, None] * poisson.pmf(goals, away)[None, :]
            for line in [-2, -1, -0.75, -0.25, 0, 0.5, 1, 1.25, 2]:
                with self.subTest(home=home, away=away, line=line):
                    diff = goals[:, None] + line - goals[None, :]
                    expected = [matrix[diff > 0].sum(), matrix[diff == 0].sum(), matrix[diff < 0].sum()]
                    np.testing.assert_allclose(calculate_prob_handicap(home, away, line), expected, atol=1e-12)

    def test_invalid_inputs(self):
        for args in [(-1, 1, 0), (1, np.nan, 0), (1, 1, np.nan), (np.inf, 1, 0)]:
            with self.subTest(args=args), self.assertRaises(ValueError):
                calculate_prob_handicap(*args)

    def test_csv_export_and_missing_handicap(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            input_path = root / "predictions.csv"
            handicap_path = root / "handicap.csv"
            output_path = root / "output.csv"
            pd.DataFrame({
                "match_id": [3, 1, 2], "pred_home_xg": [0.1, 1.3, 2.0],
                "pred_away_xg": [0.1, 1.6, 0.8], "other": ["a", "b", "c"],
            }).to_csv(input_path, index=False)
            pd.DataFrame({"match_id": [1, 2, 3], "rq": [0.25, -1, 0]}).to_csv(handicap_path, index=False)
            script = Path(__file__).with_name("lambda_to_handicap.py")
            command = [
                sys.executable, "-X", "utf8", "-s", str(script), "--input", str(input_path),
                "--handicap-data", str(handicap_path), "--output", str(output_path),
            ]
            result = subprocess.run(command, capture_output=True, text=True, encoding="utf-8")
            self.assertEqual(result.returncode, 0, result.stderr)
            out = pd.read_csv(output_path)
            self.assertEqual(out["match_id"].tolist(), [3, 1, 2])
            self.assertEqual(out["rq"].tolist(), [0, 0.25, -1])
            np.testing.assert_allclose(out[["home_prob", "draw_prob", "away_prob"]].sum(axis=1), 1)
            self.assertEqual(out.loc[1, "draw_prob"], 0)
            self.assertEqual(out.loc[0, "predicted_result"], "Draw")
            expected_labels = np.array(["HomeWin", "Draw", "AwayWin"])[
                out[["home_prob", "draw_prob", "away_prob"]].to_numpy().argmax(axis=1)
            ]
            np.testing.assert_array_equal(out["predicted_result"], expected_labels)
            pd.DataFrame({"match_id": [1], "rq": [0]}).to_csv(handicap_path, index=False)
            result = subprocess.run(command, capture_output=True, text=True, encoding="utf-8")
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("Handicap data missing", result.stderr)


if __name__ == "__main__":
    unittest.main()

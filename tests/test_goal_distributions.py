import unittest

import numpy as np
import torch
from scipy.stats import poisson

from src.models.football_tft import build_football_tft
from src.training.losses import PoissonGoalsLoss


class GoalDistributionTest(unittest.TestCase):
    def test_poisson_loss_matches_log_likelihood_and_backpropagates(self):
        rates = torch.tensor([[1.5, 0.8], [2.4, 1.1]], dtype=torch.float64, requires_grad=True)
        targets = torch.tensor([[0.0, 3.0], [2.0, 1.0]], dtype=torch.float64)
        loss = PoissonGoalsLoss()(rates, targets)
        expected = -poisson.logpmf(targets.numpy(), rates.detach().numpy()).sum(axis=1).mean()
        self.assertAlmostEqual(loss.item(), expected, places=10)
        loss.backward()
        np.testing.assert_allclose(rates.grad.numpy(), (1 - targets.numpy() / rates.detach().numpy()) / 2)

    def test_legacy_checkpoint_defaults_to_negative_binomial(self):
        schema = {
            "observed_history_columns": ["a", "b"],
            "static_numeric_columns": ["rank"],
            "static_categorical_columns": [],
            "category_cardinalities": [],
        }
        config = {"hidden_dim": 4, "num_heads": 1, "lstm_layers": 1, "dropout": 0.0}
        model = build_football_tft(schema, config)
        self.assertEqual(model.distribution, "negative_binomial")
        self.assertEqual(model.distribution_head.out_features, 4)
        model = build_football_tft(schema, {**config, "distribution": "poisson"})
        self.assertEqual(model.distribution_head.out_features, 2)
        with self.assertRaises(ValueError):
            build_football_tft(schema, {**config, "distribution": "unknown"})


if __name__ == "__main__":
    unittest.main()

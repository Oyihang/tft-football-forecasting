import unittest

import torch

from src.models.football_tft import FootballTFT


class LSTMStateTest(unittest.TestCase):
    def test_unknown_static_category_starts_neutral(self):
        model = FootballTFT(
            history_feature_count=2,
            static_numeric_names=[],
            static_categorical_names=["venue"],
            static_category_cardinalities=[3],
            hidden_dim=4,
            num_heads=1,
        )
        embedding = model.tft.embeddings.static_covariates_cat_emb.embed_lookup_table[0]
        self.assertTrue(torch.equal(embedding.weight[0], torch.zeros(4)))
        self.assertFalse(torch.equal(embedding.weight[1], torch.zeros(4)))
        embedding(torch.tensor([0])).sum().backward()
        self.assertTrue(torch.equal(embedding.weight.grad[0], torch.ones(4)))

    def test_multilayer_static_state_is_contiguous(self):
        devices = ["cpu"]
        if torch.cuda.is_available():
            devices.append("cuda")

        for device in devices:
            with self.subTest(device=device):
                model = FootballTFT(
                    history_feature_count=2,
                    static_numeric_names=["static"],
                    static_categorical_names=[],
                    static_category_cardinalities=[],
                    hidden_dim=4,
                    num_heads=1,
                    lstm_layers=2,
                    dropout=0,
                ).to(device)
                states = []
                handle = model.tft.encoder_lstm.register_forward_pre_hook(
                    lambda module, inputs: states.append(inputs[1])
                )
                home = torch.randn(2, 3, 2, device=device)
                away = torch.randn(2, 3, 2, device=device)
                known = torch.randn(2, 4, 1, device=device)
                static = torch.randn(2, 1, device=device)
                output = model(home, away, known, static)
                output.sum().backward()
                handle.remove()

                self.assertEqual(len(states), 1)
                self.assertTrue(all(state.is_contiguous() for state in states[0]))
                self.assertIsNotNone(model.tft.static_covariate_encoder.context_state_h.fc1.weight.grad)
                self.assertIsNotNone(model.tft.static_covariate_encoder.context_state_c.fc1.weight.grad)


if __name__ == "__main__":
    unittest.main()

"""Two-branch home/away encoder, gated fusion, and TFT count-distribution head."""
from __future__ import annotations
import torch
from torch import nn
import torch.nn.functional as F
from .tft import TemporalFusionTransformer

class FootballTFT(nn.Module):
    def __init__(self, history_feature_count: int, static_numeric_names: list[str],
                 static_categorical_names: list[str], static_category_cardinalities: list[int],
                 hidden_dim: int = 64, num_heads: int = 4, lstm_layers: int = 2,
                 dropout: float = 0.1, gru_layers: int = 1, gru_dropout: float = 0.0,
                 distribution: str = "negative_binomial"):
        super().__init__()
        if distribution not in ("poisson", "negative_binomial"):
            raise ValueError(f"Unsupported goal distribution: {distribution!r}")
        if gru_layers < 1:
            raise ValueError("gru_layers must be at least 1.")
        if not 0 <= gru_dropout < 1 or (gru_layers == 1 and gru_dropout != 0):
            raise ValueError("gru_dropout must be in [0, 1) and requires gru_layers > 1.")
        self.hidden_dim = hidden_dim
        self.distribution = distribution
        # Independent encoders: same input semantics, separate learned parameters.
        self.home_encoder = nn.GRU(history_feature_count, hidden_dim, batch_first=True,
                                   num_layers=gru_layers, dropout=gru_dropout)
        self.away_encoder = nn.GRU(history_feature_count, hidden_dim, batch_first=True,
                                   num_layers=gru_layers, dropout=gru_dropout)
        self.home_norm = nn.LayerNorm(hidden_dim)
        self.away_norm = nn.LayerNorm(hidden_dim)
        self.fusion_gate = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.Sigmoid(),
        )
        self.fusion_projection = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
        )
        self.fused_norm = nn.LayerNorm(hidden_dim)

        # The fused representation is passed into the TFT as H observed variables.
        # A one-channel history_position is the only known-in-advance time feature.
        self.tft = TemporalFusionTransformer(
            horizon_length=1,
            hidden_dim=hidden_dim,
            known_inputs=["history_position"],
            observed_inputs=[f"fused_{i}" for i in range(hidden_dim)],
            static_covariates=static_numeric_names + static_categorical_names,
            known_inputs_categorical_cardinalities=[],
            observed_inputs_categorical_cardinalities=[],
            static_covariates_categorical_cardinalities=static_category_cardinalities,
            num_heads=num_heads,
            dropout=dropout,
            lstm_layers=lstm_layers,
            quantiles=None,
        )
        # Category 0 denotes an unseen category and may never occur during training.
        if self.tft.embeddings.static_covariates_cat_emb is not None:
            with torch.no_grad():
                for embedding in self.tft.embeddings.static_covariates_cat_emb.embed_lookup_table:
                    embedding.weight[0].zero_()
        self.distribution_head = nn.Linear(hidden_dim, 2 if distribution == "poisson" else 4)

    def forward(self, home_history, away_history, known_inputs, static_covariates):
        home, _ = self.home_encoder(home_history)
        away, _ = self.away_encoder(away_history)
        home = self.home_norm(home)
        away = self.away_norm(away)

        gate = self.fusion_gate(torch.cat([home, away], dim=-1))
        # Complementary gate: the two sides compete for contribution at each step.
        fused = gate * home + (1.0 - gate) * away
        pair_interaction = self.fusion_projection(torch.cat([home, away], dim=-1))
        fused = self.fused_norm(fused + pair_interaction)

        # TFT expects a separate feature axis; H fused channels are selected by its VSN.
        latent = self.tft(
            observed_inputs=fused,
            known_inputs=known_inputs,
            static_covariates=static_covariates,
            return_hidden=True,
        )
        raw = self.distribution_head(latent[:, -1, :])
        if self.distribution == "poisson":
            return F.softplus(raw) + 1e-4
        # Mean > 0; alpha > 0. NB variance = mu + alpha * mu^2.
        mu_home = F.softplus(raw[:, 0]) + 1e-4
        alpha_home = F.softplus(raw[:, 1]) + 1e-4
        mu_away = F.softplus(raw[:, 2]) + 1e-4
        alpha_away = F.softplus(raw[:, 3]) + 1e-4
        return torch.stack([mu_home, alpha_home, mu_away, alpha_away], dim=-1)


def build_football_tft(schema: dict, model_config: dict) -> FootballTFT:
    """Build the same model from prepared-data and checkpoint metadata."""
    return FootballTFT(
        history_feature_count=len(schema["observed_history_columns"]),
        static_numeric_names=schema["static_numeric_columns"],
        static_categorical_names=schema["static_categorical_columns"],
        static_category_cardinalities=schema["category_cardinalities"],
        hidden_dim=model_config["hidden_dim"],
        num_heads=model_config["num_heads"],
        lstm_layers=model_config["lstm_layers"],
        dropout=model_config["dropout"],
        gru_layers=model_config.get("gru_layers", 1),
        gru_dropout=model_config.get("gru_dropout", 0.0),
        distribution=model_config.get("distribution", "negative_binomial"),
    )

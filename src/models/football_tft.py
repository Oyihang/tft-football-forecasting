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
                 distribution: str = "negative_binomial", observed_numeric_count: int | None = None,
                 observed_category_cardinalities: list[int] | None = None,
                 known_input_names: list[str] | None = None,
                 known_category_cardinalities: list[int] | None = None,
                 horizon_length: int = 1):
        super().__init__()
        if distribution not in ("poisson", "negative_binomial"):
            raise ValueError(f"Unsupported goal distribution: {distribution!r}")
        if gru_layers < 1:
            raise ValueError("gru_layers must be at least 1.")
        if not 0 <= gru_dropout < 1 or (gru_layers == 1 and gru_dropout != 0):
            raise ValueError("gru_dropout must be in [0, 1) and requires gru_layers > 1.")
        self.hidden_dim = hidden_dim
        self.distribution = distribution
        self.horizon_length = horizon_length
        self.observed_category_cardinalities = observed_category_cardinalities or []
        self.observed_numeric_count = (
            history_feature_count - len(self.observed_category_cardinalities)
            if observed_numeric_count is None else observed_numeric_count
        )
        if self.observed_numeric_count + len(self.observed_category_cardinalities) != history_feature_count:
            raise ValueError("Observed feature counts do not match history_feature_count.")
        self.history_category_embeddings = nn.ModuleList([
            nn.Embedding(cardinality, hidden_dim, padding_idx=0)
            for cardinality in self.observed_category_cardinalities
        ])
        gru_input_count = self.observed_numeric_count + hidden_dim * len(self.observed_category_cardinalities)
        # Independent encoders: same input semantics, separate learned parameters.
        self.home_encoder = nn.GRU(gru_input_count, hidden_dim, batch_first=True,
                                   num_layers=gru_layers, dropout=gru_dropout)
        self.away_encoder = nn.GRU(gru_input_count, hidden_dim, batch_first=True,
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
        # Historical-position and match-context covariates are supplied separately.
        self.tft = TemporalFusionTransformer(
            horizon_length=horizon_length,
            hidden_dim=hidden_dim,
            known_inputs=known_input_names or ["history_position"],
            observed_inputs=[f"fused_{i}" for i in range(hidden_dim)],
            static_covariates=static_numeric_names + static_categorical_names,
            known_inputs_categorical_cardinalities=known_category_cardinalities or [],
            observed_inputs_categorical_cardinalities=[],
            static_covariates_categorical_cardinalities=static_category_cardinalities,
            num_heads=num_heads,
            dropout=dropout,
            lstm_layers=lstm_layers,
            quantiles=None,
        )
        # Category 0 denotes an unseen category and may never occur during training.
        for categorical_embedding in (
            self.tft.embeddings.known_inputs_cat_emb,
            self.tft.embeddings.static_covariates_cat_emb,
        ):
            if categorical_embedding is None:
                continue
            with torch.no_grad():
                for embedding in categorical_embedding.embed_lookup_table:
                    embedding.weight[0].zero_()
        self.distribution_head = nn.Linear(hidden_dim, 2 if distribution == "poisson" else 4)

    def _embed_history_categories(self, history):
        if history.shape[-1] != self.observed_numeric_count + len(self.observed_category_cardinalities):
            raise ValueError(
                "History tensor feature count does not match the observed feature schema."
            )
        if not self.history_category_embeddings:
            return history
        numeric = history[..., :self.observed_numeric_count]
        categorical = history[..., self.observed_numeric_count:]
        if not torch.equal(categorical, categorical.long().to(categorical.dtype)):
            raise ValueError("Observed categorical values must be integers.")
        embedded = []
        for index, (cardinality, embedding) in enumerate(zip(
            self.observed_category_cardinalities, self.history_category_embeddings
        )):
            values = categorical[..., index].long()
            if values.numel() and (values.min() < 0 or values.max() >= cardinality):
                raise ValueError(
                    f"Observed categorical variable {index} has values outside [0, {cardinality - 1}]."
                )
            embedded.append(embedding(values))
        return torch.cat([numeric, *embedded], dim=-1)

    def forward(self, home_history, away_history, known_inputs, static_covariates):
        home_history = self._embed_history_categories(home_history)
        away_history = self._embed_history_categories(away_history)
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
    observed_columns = schema["observed_history_columns"]
    observed_numeric_columns = schema.get("observed_numeric_columns", observed_columns)
    return FootballTFT(
        history_feature_count=len(observed_columns),
        static_numeric_names=schema["static_numeric_columns"],
        static_categorical_names=schema["static_categorical_columns"],
        static_category_cardinalities=schema.get(
            "static_category_cardinalities", schema.get("category_cardinalities", [])
        ),
        hidden_dim=model_config["hidden_dim"],
        num_heads=model_config["num_heads"],
        lstm_layers=model_config["lstm_layers"],
        dropout=model_config["dropout"],
        gru_layers=model_config.get("gru_layers", 1),
        gru_dropout=model_config.get("gru_dropout", 0.0),
        distribution=model_config.get("distribution", "negative_binomial"),
        observed_numeric_count=len(observed_numeric_columns),
        observed_category_cardinalities=schema.get("observed_category_cardinalities", []),
        known_input_names=schema.get("known_inputs", ["history_position"]),
        known_category_cardinalities=schema.get("known_category_cardinalities", []),
        horizon_length=schema.get("horizon_length", 1),
    )

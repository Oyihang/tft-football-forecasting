"""Count-distribution likelihoods for home/away goal counts."""
import torch
from torch import nn


class PoissonGoalsLoss(nn.Module):
    """Sum the two Poisson negative log-likelihoods, then average over matches."""
    def forward(self, rates, targets):
        return -torch.distributions.Poisson(rates).log_prob(targets).sum(dim=-1).mean()


class NegativeBinomialGoalsLoss(nn.Module):
    """Independent NB likelihoods for home and away goals."""
    def forward(self, params, targets):
        mu_h, alpha_h, mu_a, alpha_a = params.unbind(dim=-1)
        y_h, y_a = targets.unbind(dim=-1)
        eps = 1e-6
        def nll(y, mu, alpha):
            alpha = alpha.clamp(min=eps, max=20.0)
            mu = mu.clamp(min=eps, max=100.0)
            total_count = (1.0 / alpha).clamp(min=eps, max=1e6)
            logits = torch.log((mu * alpha).clamp(min=eps, max=1e8))
            dist = torch.distributions.NegativeBinomial(total_count=total_count, logits=logits)
            return -dist.log_prob(y)
        return (nll(y_h, mu_h, alpha_h) + nll(y_a, mu_a, alpha_a)).mean()

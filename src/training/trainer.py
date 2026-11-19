from __future__ import annotations
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from .losses import NegativeBinomialGoalsLoss, PoissonGoalsLoss

def make_loader(home, away, known, static, targets=None, batch_size=128, shuffle=False, num_workers=0):
    arrays = [home, away, known, static]
    tensors = [torch.as_tensor(np.asarray(a), dtype=torch.float32) for a in arrays]
    if targets is not None:
        tensors.append(torch.as_tensor(np.asarray(targets), dtype=torch.float32))
    return DataLoader(TensorDataset(*tensors), batch_size=batch_size, shuffle=shuffle,
                      num_workers=num_workers, pin_memory=torch.cuda.is_available())

def run_epoch(model, loader, device, optimizer=None, grad_clip=1.0):
    training = optimizer is not None
    model.train(training)
    if model.distribution == "poisson":
        criterion = PoissonGoalsLoss()
    elif model.distribution == "negative_binomial":
        criterion = NegativeBinomialGoalsLoss()
    else:
        raise ValueError(f"Unsupported goal distribution: {model.distribution!r}")
    total_loss, n = 0.0, 0
    preds, targets = [], []
    context = torch.enable_grad() if training else torch.no_grad()
    with context:
        for home, away, known, static, y in loader:
            home, away, known, static, y = [x.to(device, non_blocking=True) for x in (home, away, known, static, y)]
            if training:
                optimizer.zero_grad(set_to_none=True)
            params = model(home, away, known, static)
            loss = criterion(params, y)
            if training:
                loss.backward()
                if grad_clip and grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                optimizer.step()
            batch_n = len(y)
            total_loss += float(loss.detach()) * batch_n
            n += batch_n
            means = params if model.distribution == "poisson" else params[:, (0, 2)]
            preds.append(means.detach().cpu().numpy())
            targets.append(y.detach().cpu().numpy())
    return total_loss/max(n,1), np.concatenate(preds), np.concatenate(targets)

@torch.no_grad()
def predict_params(model, loader, device):
    model.eval()
    chunks=[]
    for home, away, known, static, *rest in loader:
        params=model(home.to(device),away.to(device),known.to(device),static.to(device))
        chunks.append(params.cpu().numpy())
    output_dim = 2 if model.distribution == "poisson" else 4
    return np.concatenate(chunks) if chunks else np.empty((0, output_dim), np.float32)

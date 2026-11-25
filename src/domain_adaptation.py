"""Unsupervised Domain Adaptation modules (Multi-Kernel MMD & DANN) for cross-subject SEED."""

from __future__ import annotations

import math
from typing import Sequence

import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Function
from torch.utils.data import DataLoader, TensorDataset


class _GradReverseFn(Function):
    @staticmethod
    def forward(ctx, x: torch.Tensor, lambd: float) -> torch.Tensor:
        ctx.lambd = lambd
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        return -ctx.lambd * grad_output, None


def grad_reverse(x: torch.Tensor, lambd: float = 1.0) -> torch.Tensor:
    return _GradReverseFn.apply(x, lambd)


class MultiKernelMMDLoss(nn.Module):
    """Multi-kernel Gaussian RBF Maximum Mean Discrepancy (MMD) loss."""

    def __init__(self, kernel_mul: float = 2.0, kernel_num: int = 5) -> None:
        super().__init__()
        self.kernel_mul = kernel_mul
        self.kernel_num = kernel_num

    def _gaussian_kernel(self, source: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        total = torch.cat([source, target], dim=0)
        dist_sq = torch.cdist(total, total, p=2.0) ** 2
        n_total = total.shape[0]
        bandwidth = dist_sq.sum() / max(n_total * n_total - n_total, 1)
        bandwidth = bandwidth.detach().clamp(min=1e-6)
        bandwidth /= self.kernel_mul ** (self.kernel_num // 2)

        kernels = 0.0
        for i in range(self.kernel_num):
            bw_i = bandwidth * (self.kernel_mul ** i)
            kernels = kernels + torch.exp(-dist_sq / bw_i)
        return kernels

    def forward(self, source: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        n_s = source.shape[0]
        n_t = target.shape[0]
        if n_s < 2 or n_t < 2:
            return torch.tensor(0.0, device=source.device)
        K = self._gaussian_kernel(source, target)
        K_ss = K[:n_s, :n_s]
        K_tt = K[n_s:, n_s:]
        K_st = K[:n_s, n_s:]
        return (K_ss.mean() + K_tt.mean() - 2.0 * K_st.mean()).clamp(min=0.0)


class DomainDiscriminator(nn.Module):
    """Domain classifier for Domain-Adversarial Neural Network (DANN)."""

    def __init__(self, latent_dim: int = 64, dropout: float = 0.3) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim, latent_dim),
            nn.BatchNorm1d(latent_dim),
            nn.ELU(),
            nn.Dropout(dropout),
            nn.Linear(latent_dim, 2),
        )

    def forward(self, z: torch.Tensor, lambd: float = 1.0) -> torch.Tensor:
        return self.net(grad_reverse(z, lambd))


def train_pytorch_classifier(
    model: nn.Module,
    X_source: np.ndarray,
    y_source: np.ndarray,
    X_target_unlabeled: np.ndarray | None = None,
    da_mode: str = "none",
    epochs: int = 40,
    batch_size: int = 128,
    lr: float = 1e-3,
    weight_decay: float = 5e-4,
    mmd_weight: float = 0.5,
    dann_weight: float = 0.3,
    latent_dim: int = 64,
) -> nn.Module:
    """Train a PyTorch emotion model with optional MMD and/or DANN domain adaptation.

    Strictly uses ONLY unlabeled target features X_target_unlabeled when da_mode != 'none'.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    use_mmd = da_mode in ("mmd", "mmd_dann") and X_target_unlabeled is not None
    use_dann = da_mode in ("dann", "mmd_dann") and X_target_unlabeled is not None

    mmd_loss_fn = MultiKernelMMDLoss().to(device) if use_mmd else None
    domain_disc = DomainDiscriminator(latent_dim=latent_dim).to(device) if use_dann else None

    # Dedicated higher LR and zero weight decay for learnable graph adjacency (Song et al., 2018)
    adj_params = [p for n, p in model.named_parameters() if n == "adj"]
    other_params = [p for n, p in model.named_parameters() if n != "adj"]
    param_groups = [{"params": other_params, "lr": lr, "weight_decay": weight_decay}]
    if adj_params:
        param_groups.append({"params": adj_params, "lr": lr * 5.0, "weight_decay": 0.0})
    if domain_disc is not None:
        param_groups.append({"params": list(domain_disc.parameters()), "lr": lr, "weight_decay": weight_decay})
    params = list(model.parameters()) + (list(domain_disc.parameters()) if domain_disc is not None else [])
    optimizer = torch.optim.AdamW(param_groups)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    cls_criterion = nn.CrossEntropyLoss(label_smoothing=0.05)
    dom_criterion = nn.CrossEntropyLoss()

    src_loader = DataLoader(
        TensorDataset(torch.from_numpy(X_source), torch.from_numpy(y_source.astype(np.int64))),
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
    )
    tgt_loader = None
    if use_mmd or use_dann:
        tgt_loader = DataLoader(
            TensorDataset(torch.from_numpy(X_target_unlabeled)),
            batch_size=batch_size,
            shuffle=True,
            drop_last=True,
        )

    for epoch in range(epochs):
        model.train()
        if domain_disc is not None:
            domain_disc.train()

        p = float(epoch) / max(epochs - 1, 1)
        grl_lambda = 2.0 / (1.0 + math.exp(-10.0 * p)) - 1.0

        tgt_iter = iter(tgt_loader) if tgt_loader is not None else None

        for xb_s, yb_s in src_loader:
            xb_s = xb_s.to(device)
            yb_s = yb_s.to(device)

            z_s = model.encode(xb_s)
            logits_s = model.classifier(z_s)
            loss = cls_criterion(logits_s, yb_s)
            if hasattr(model, "adj"):
                loss = loss + 1e-4 * torch.abs(model.adj).mean()

            if tgt_iter is not None:
                try:
                    (xb_t,) = next(tgt_iter)
                except StopIteration:
                    tgt_iter = iter(tgt_loader)
                    (xb_t,) = next(tgt_iter)
                xb_t = xb_t.to(device)
                z_t = model.encode(xb_t)

                if use_mmd and mmd_loss_fn is not None:
                    loss = loss + mmd_weight * mmd_loss_fn(z_s, z_t)

                if use_dann and domain_disc is not None:
                    dom_s = domain_disc(z_s, lambd=grl_lambda)
                    dom_t = domain_disc(z_t, lambd=grl_lambda)
                    y_dom_s = torch.zeros(len(z_s), dtype=torch.long, device=device)
                    y_dom_t = torch.ones(len(z_t), dtype=torch.long, device=device)
                    loss_d = 0.5 * (dom_criterion(dom_s, y_dom_s) + dom_criterion(dom_t, y_dom_t))
                    loss = loss + dann_weight * loss_d

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, max_norm=5.0)
            optimizer.step()

        scheduler.step()

    return model


@torch.no_grad()
def predict_pytorch_classifier(model: nn.Module, X: np.ndarray, batch_size: int = 256) -> np.ndarray:
    device = next(model.parameters()).device
    model.eval()
    preds = []
    for s in range(0, len(X), batch_size):
        xb = torch.from_numpy(X[s : s + batch_size]).to(device)
        preds.append(model(xb).argmax(dim=1).cpu().numpy())
    return np.concatenate(preds, axis=0)

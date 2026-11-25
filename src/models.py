"""Emotion recognition models on SEED 62-channel graph features:
1. SVM (LinearSVC / RBF-SVC baseline)
2. MLP (3-layer PyTorch baseline)
3. TSception (Multi-scale spectral-spatial asymmetric CNN baseline)
4. DGCNN (Song et al., 2018: Dynamic Graph Convolutional Neural Network with
   Chebyshev spectral graph convolution and learnable adjacency matrix A*)
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC


class SVMClassifier:
    """Fast linear SVM baseline on flattened (62 * F) EEG features."""

    def __init__(self, c: float = 1.0) -> None:
        self.scaler = StandardScaler()
        self.clf = LinearSVC(C=c, max_iter=3000, dual="auto", random_state=42)

    def fit(self, X: np.ndarray, y: np.ndarray) -> "SVMClassifier":
        Xf = X.reshape(len(X), -1)
        Xf = self.scaler.fit_transform(Xf)
        self.clf.fit(Xf, y)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        Xf = self.scaler.transform(X.reshape(len(X), -1))
        return self.clf.predict(Xf)


class MLPNet(nn.Module):
    """3-layer MLP baseline on flattened (62 * F) features."""

    def __init__(self, n_nodes: int = 62, in_features: int = 10, latent_dim: int = 64, n_classes: int = 3, dropout: float = 0.3) -> None:
        super().__init__()
        in_dim = n_nodes * in_features
        self.encoder = nn.Sequential(
            nn.Linear(in_dim, 256),
            nn.BatchNorm1d(256),
            nn.ELU(),
            nn.Dropout(dropout),
            nn.Linear(256, latent_dim),
            nn.BatchNorm1d(latent_dim),
            nn.ELU(),
            nn.Dropout(dropout),
        )
        self.classifier = nn.Linear(latent_dim, n_classes)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x.flatten(1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.encode(x))


class TSceptionNet(nn.Module):
    """Multi-scale spectral-spatial CNN inspired by TSception (Ding et al., 2022).

    Captures multi-scale feature band interactions and asymmetric hemispheric spatial patterns.
    """

    def __init__(self, n_nodes: int = 62, in_features: int = 10, latent_dim: int = 64, n_classes: int = 3, dropout: float = 0.3) -> None:
        super().__init__()
        # Multi-scale feature-axis convolutions (kernel widths 1, 3, 5 across spectral bands)
        self.branch1 = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=(1, 1)),
            nn.BatchNorm2d(16),
            nn.LeakyReLU(0.1),
        )
        self.branch2 = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=(1, 3), padding=(0, 1)),
            nn.BatchNorm2d(16),
            nn.LeakyReLU(0.1),
        )
        self.branch3 = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=(1, 5), padding=(0, 2)),
            nn.BatchNorm2d(16),
            nn.LeakyReLU(0.1),
        )
        # Spatial asymmetric hemispheric + global electrode convs
        self.spatial_hemi = nn.Sequential(
            nn.Conv2d(48, 32, kernel_size=(n_nodes // 2, 1), stride=(n_nodes // 2, 1)),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.1),
        )
        self.spatial_global = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=(2, 1)),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.1),
        )
        self.fusion = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(32 * in_features, latent_dim),
            nn.BatchNorm1d(latent_dim),
            nn.ELU(),
        )
        self.classifier = nn.Linear(latent_dim, n_classes)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, 62, F) -> (B, 1, 62, F)
        x = x.unsqueeze(1)
        m = torch.cat([self.branch1(x), self.branch2(x), self.branch3(x)], dim=1)
        s = self.spatial_global(self.spatial_hemi(m))
        return self.fusion(s)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.encode(x))


class ChebGraphConv(nn.Module):
    """Chebyshev spectral graph convolution of order K (Defferrard et al., 2016)."""

    def __init__(self, in_features: int, out_features: int, cheb_k: int = 2) -> None:
        super().__init__()
        self.cheb_k = int(cheb_k)
        self.weight = nn.Parameter(torch.empty(cheb_k, in_features, out_features))
        self.bias = nn.Parameter(torch.zeros(out_features))
        nn.init.xavier_uniform_(self.weight)

    @staticmethod
    def _normalized_laplacian(adj: torch.Tensor) -> torch.Tensor:
        # Enforce symmetric non-negative adjacency with self-loops
        A = F.relu(0.5 * (adj + adj.T))
        N = A.shape[0]
        I = torch.eye(N, device=A.device, dtype=A.dtype)
        A = A + I
        deg = A.sum(dim=1).clamp(min=1e-6)
        d_inv_sqrt = deg.pow(-0.5)
        L_sym = I - (d_inv_sqrt[:, None] * A * d_inv_sqrt[None, :])
        # Rescale eigenvalues to [-1, 1] for Chebyshev polynomials: L_tilde = L_sym - I
        return L_sym - I

    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        """x: (B, N, F_in), adj: (N, N) -> (B, N, F_out)."""
        L_tilde = self._normalized_laplacian(adj)
        Tx_0 = x
        out = torch.einsum("bni,io->bno", Tx_0, self.weight[0])
        if self.cheb_k > 1:
            Tx_1 = torch.einsum("nm,bmf->bnf", L_tilde, x)
            out = out + torch.einsum("bni,io->bno", Tx_1, self.weight[1])
            for k in range(2, self.cheb_k):
                Tx_2 = 2.0 * torch.einsum("nm,bmf->bnf", L_tilde, Tx_1) - Tx_0
                out = out + torch.einsum("bni,io->bno", Tx_2, self.weight[k])
                Tx_0, Tx_1 = Tx_1, Tx_2
        return out + self.bias


class DGCNNNet(nn.Module):
    """Dynamic Graph Convolutional Neural Network (Song et al., IEEE TAFFC 2018).

    Initializes adjacency matrix A from 3D electrode spatial geometry and jointly
    optimizes A via backpropagation to adaptively discover emotion-relevant brain connectivity.
    """

    def __init__(
        self,
        init_adj: np.ndarray,
        in_features: int = 10,
        hidden_dim: int = 64,
        latent_dim: int = 64,
        cheb_k: int = 2,
        n_classes: int = 3,
        dropout: float = 0.3,
    ) -> None:
        super().__init__()
        n_nodes = init_adj.shape[0]
        node_out = 8
        self.adj = nn.Parameter(torch.tensor(init_adj, dtype=torch.float32))
        self.bn_in = nn.BatchNorm1d(n_nodes)
        self.gcn1 = ChebGraphConv(in_features, hidden_dim // 2, cheb_k=cheb_k)
        self.bn1 = nn.BatchNorm1d(n_nodes)
        self.node_compress = nn.Linear(hidden_dim // 2, node_out)
        self.res_proj = nn.Linear(in_features, node_out, bias=False)

        self.readout = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.5),
            nn.Linear(n_nodes * node_out, latent_dim),
            nn.BatchNorm1d(latent_dim),
            nn.ELU(),
        )
        self.classifier = nn.Linear(latent_dim, n_classes)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, 62, F)
        h0 = self.bn_in(x)
        h = F.elu(self.bn1(self.gcn1(h0, self.adj)))
        h = F.elu(self.node_compress(h) + self.res_proj(h0))
        return self.readout(h)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.encode(x))

    def get_learned_adjacency(self) -> np.ndarray:
        with torch.no_grad():
            A = F.relu(0.5 * (self.adj + self.adj.T)).detach().cpu().numpy()
        return A

"""Torch network architectures for GANs, surrogate classifier, and 1D CNN victim."""
from __future__ import annotations
import torch
from torch import nn
from torch.nn import functional as F


class Generator(nn.Module):
    def __init__(self, z_dim: int, n_numeric: int, categorical_sizes: list[int]):
        super().__init__()
        self.z_dim = int(z_dim)
        self.n_numeric = int(n_numeric)
        self.categorical_sizes = list(categorical_sizes)
        out_dim = self.n_numeric + sum(categorical_sizes)
        self.layers = nn.Sequential(
            nn.Linear(z_dim, 128), nn.ReLU(), nn.Linear(128, 256), nn.ReLU(),
            nn.Linear(256, 128), nn.ReLU(), nn.Linear(128, out_dim)
        )

    def forward(self, noise: torch.Tensor, *, temperature: float = 0.75,
                stochastic: bool = True) -> torch.Tensor:
        logits = self.layers(noise)
        features = []
        if self.n_numeric:
            features.append(torch.sigmoid(logits[:, :self.n_numeric]))
        offset = self.n_numeric
        for num_classes in self.categorical_sizes:
            group = logits[:, offset:offset+num_classes]
            if stochastic:
                encoded = F.gumbel_softmax(group, tau=temperature, hard=True, dim=1)
            else:
                encoded = F.one_hot(group.argmax(dim=1), num_classes).to(group.dtype)
            features.append(encoded)
            offset += num_classes
        return torch.cat(features, dim=1)


class Discriminator(nn.Module):
    """Outputs logits, not probabilities (for stable BCEWithLogitsLoss)."""
    def __init__(self, input_dim: int):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_dim, 128), nn.LeakyReLU(0.2),
            nn.Linear(128, 256), nn.LeakyReLU(0.2),
            nn.Linear(256, 128), nn.LeakyReLU(0.2),
            nn.Linear(128, 1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x).flatten()


class SurrogateMLP(nn.Module):
    def __init__(self, n_features: int, n_classes: int):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(n_features, 128), nn.ReLU(),
            nn.Linear(128, 64), nn.ReLU()
        )
        self.classifier = nn.Linear(64, n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.encoder(x))

    def embedding(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)


class CNN1D(nn.Module):
    """Documented implementation choice; paper does not specify CNN layers."""
    def __init__(self, n_classes: int):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=3, padding=1), nn.ReLU(),
            nn.Conv1d(32, 64, kernel_size=3, padding=1), nn.ReLU(),
            nn.AdaptiveAvgPool1d(1), nn.Flatten()
        )
        self.head = nn.Sequential(nn.Linear(64, 64), nn.ReLU(), nn.Linear(64, n_classes))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.conv(x.unsqueeze(1)))

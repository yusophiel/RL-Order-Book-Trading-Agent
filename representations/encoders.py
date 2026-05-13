"""
Compress the raw LOB (original order book vector) into a shorter latent representation.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn


# Base class
class BaseEncoder:
    def __init__(self, input_dim: int, latent_dim: int):
        self.input_dim = input_dim
        self.latent_dim = latent_dim

    # Input a raw LOB vector, output a latent vector
    def encode(self, raw: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    # Train the encoder using a batch of raw LOB snapshots
    def fit(self, dataset: np.ndarray, n_epochs: int = 50, lr: float = 1e-3):
        raise NotImplementedError


# MLP encoder
class _MLPNet(nn.Module):
    def __init__(self, in_dim: int, hidden: int, latent: int):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, latent),
        )

    def forward(self, x):
        return self.encoder(x)


# Learn a low-dimensional compressed representation that preserves the key information of the raw LOB using MLP.
class MLPEncoder(BaseEncoder):

    def __init__(self, input_dim: int, hidden_dim: int = 64, latent_dim: int = 8):
        super().__init__(input_dim, latent_dim)
        self.net = _MLPNet(input_dim, hidden_dim, latent_dim)
        self._decoder = nn.Linear(latent_dim, input_dim)
        self._trained = False

    def encode(self, raw: np.ndarray) -> np.ndarray:
        if not self._trained:
            # Return random-projection until trained
            return np.tanh(raw[:self.latent_dim] if len(raw) >= self.latent_dim
                           else np.pad(raw, (0, self.latent_dim - len(raw))))
        x = torch.tensor(raw, dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            return self.net(x).squeeze(0).numpy()

    def fit(self, dataset: np.ndarray, n_epochs: int = 50, lr: float = 1e-3):
        X = torch.tensor(dataset, dtype=torch.float32)
        optim = torch.optim.Adam(
            list(self.net.parameters()) + list(self._decoder.parameters()), lr=lr
        )
        for _ in range(n_epochs):
            z = self.net(X)
            recon = self._decoder(z)
            loss = nn.functional.mse_loss(recon, X)
            optim.zero_grad()
            loss.backward()
            optim.step()
        self._trained = True


# CNN encoder
# Treating LOB as a one-dimensional structured signal: 4 channels × lob_depth price levels
# It can learn the local structure between different levels of the order book,
# such as the relationship between the first and second tiers.
class _CNNNet(nn.Module):
    def __init__(self, lob_depth: int, latent: int):
        super().__init__()
        # Treat the LOB as a (4, lob_depth) 1-D signal
        self.conv = nn.Sequential(
            nn.Conv1d(4, 16, kernel_size=3, padding=1), nn.ReLU(),
            nn.Conv1d(16, 8, kernel_size=3, padding=1), nn.ReLU(),
        )
        conv_out = 8 * lob_depth
        self.fc = nn.Linear(conv_out, latent)

    def forward(self, x, lob_depth):
        # x shape: (batch, 4*lob_depth)   → (batch, 4, lob_depth)
        b = x.shape[0]
        x = x.view(b, 4, lob_depth)
        x = self.conv(x)
        x = x.flatten(1)
        return self.fc(x)


class CNNEncoder(BaseEncoder):
    def __init__(self, input_dim: int, lob_depth: int = 5, latent_dim: int = 8):
        super().__init__(input_dim, latent_dim)
        self.lob_depth = lob_depth
        # input_dim may include +1 for last_trade; we trim to 4*lob_depth
        self._lob_raw_dim = 4 * lob_depth
        self.net = _CNNNet(lob_depth, latent_dim)
        self._decoder_fc = nn.Linear(latent_dim, self._lob_raw_dim)
        self._trained = False

    def encode(self, raw: np.ndarray) -> np.ndarray:
        lob_raw = raw[:self._lob_raw_dim]
        if not self._trained:
            return np.tanh(lob_raw[:self.latent_dim])
        x = torch.tensor(lob_raw, dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            return self.net(x, self.lob_depth).squeeze(0).numpy()

    def fit(self, dataset: np.ndarray, n_epochs: int = 50, lr: float = 1e-3):
        X_full = torch.tensor(dataset[:, :self._lob_raw_dim], dtype=torch.float32)
        params = list(self.net.parameters()) + list(self._decoder_fc.parameters())
        optim = torch.optim.Adam(params, lr=lr)
        for _ in range(n_epochs):
            z = self.net(X_full, self.lob_depth)
            recon = self._decoder_fc(z)
            loss = nn.functional.mse_loss(recon, X_full)
            optim.zero_grad()
            loss.backward()
            optim.step()
        self._trained = True


# Complete symmetric autoencoder
class _AENet(nn.Module):
    def __init__(self, in_dim: int, latent: int):
        super().__init__()
        hidden = max(latent * 4, 32)
        self.encoder = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, latent), nn.Tanh(),
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent, hidden), nn.ReLU(),
            nn.Linear(hidden, in_dim),
        )

    def forward(self, x):
        z = self.encoder(x)
        return self.decoder(z), z


class AutoencoderEncoder(BaseEncoder):
    def __init__(self, input_dim: int, latent_dim: int = 8):
        super().__init__(input_dim, latent_dim)
        self.net = _AENet(input_dim, latent_dim)
        self._trained = False

    def encode(self, raw: np.ndarray) -> np.ndarray:
        if not self._trained:
            return np.zeros(self.latent_dim, dtype=np.float32)
        x = torch.tensor(raw, dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            _, z = self.net(x)
        return z.squeeze(0).numpy()

    def fit(self, dataset: np.ndarray, n_epochs: int = 50, lr: float = 1e-3):
        X = torch.tensor(dataset, dtype=torch.float32)
        optim = torch.optim.Adam(self.net.parameters(), lr=lr)
        for _ in range(n_epochs):
            recon, _ = self.net(X)
            loss = nn.functional.mse_loss(recon, X)
            optim.zero_grad()
            loss.backward()
            optim.step()
        self._trained = True


# Factory: Instantiate an encoder from a string key.
def make_encoder(encoder_type: str, input_dim: int, cfg) -> BaseEncoder:
    latent = cfg.encoder_latent
    hidden = cfg.encoder_hidden
    lob_depth = getattr(cfg, 'lob_depth', 5)

    if encoder_type == 'mlp':
        return MLPEncoder(input_dim, hidden_dim=hidden, latent_dim=latent)
    elif encoder_type == 'cnn':
        return CNNEncoder(input_dim, lob_depth=lob_depth, latent_dim=latent)
    elif encoder_type == 'autoencoder':
        return AutoencoderEncoder(input_dim, latent_dim=latent)
    else:
        raise ValueError(f"Unknown encoder type: {encoder_type!r}")

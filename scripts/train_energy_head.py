"""The energy head of scripts/train_energy.py (shared with scripts/energy_decode.py)."""
import torch
import torch.nn as nn


class EnergyHead(nn.Module):
    """causal states over the chunk (B, C, d) -> scalar energy; mean and last state, small MLP"""
    def __init__(self, d):
        super().__init__()
        self.net = nn.Sequential(nn.LayerNorm(2 * d), nn.Linear(2 * d, d), nn.GELU(), nn.Linear(d, 1))

    def forward(self, h):
        return self.net(torch.cat([h.mean(1), h[:, -1]], -1))[:, 0]

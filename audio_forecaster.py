"""Speech forecaster for imagined look-ahead (2026-11-08). A small causal transformer over streamed HuBERT features
(25 fps, each frame computed from audio up to that frame) that predicts, at every frame i, the next K frames i+1..i+K
from frames <= i only. A motion model trained with a fixed look-ahead of s frames (speech_to_motion_small.py
--any_shifts / --feat_shift) can then run with zero real look-ahead: for each motion slot, the s frames past its end are
replaced by the forecast made at its end (SmallS2M.encode_audio, model.forecaster). Trained by scripts/train_forecaster.py.
"""
import torch
import torch.nn as nn


class Forecaster(nn.Module):
    def __init__(self, K=7, d=512, layers=6, F=262, feat_mean=None, feat_std=None):
        super().__init__()
        self.K, self.d = K, d
        self.cfg = dict(K=K, d=d, layers=layers, F=F)
        self.register_buffer("feat_mean", torch.zeros(1024) if feat_mean is None else torch.as_tensor(feat_mean).float())
        self.register_buffer("feat_std", torch.ones(1024) if feat_std is None else torch.as_tensor(feat_std).float())
        self.inp = nn.Linear(1024, d)
        self.pos = nn.Parameter(torch.randn(F, d) * 0.02)
        el = nn.TransformerEncoderLayer(d, 8, 4 * d, dropout=0.1, batch_first=True, norm_first=True)
        self.enc = nn.TransformerEncoder(el, layers)
        self.out = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, K * 1024))
        self.register_buffer("mask", torch.triu(torch.full((F, F), float("-inf")), 1))

    def forward(self, feats):
        """(B, F, 1024) raw features -> (B, F, K, 1024) raw-scale forecasts of frames i+1..i+K made at frame i"""
        B, F, _ = feats.shape
        x = self.inp((feats.float() - self.feat_mean) / self.feat_std) + self.pos[:F]
        h = self.enc(x, mask=self.mask[:F, :F])
        z = self.out(h).view(B, F, self.K, 1024)
        return z * self.feat_std + self.feat_mean


def load_forecaster(path):
    st = torch.load(path, map_location="cpu")
    m = Forecaster(**st["cfg"]).cuda().eval()
    m.load_state_dict(st["state"])
    return m

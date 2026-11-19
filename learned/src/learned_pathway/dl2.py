"""DL-2 (notebook 03): a small trained attention network on frozen piece features.

A clip is a *bag* of pieces, each described by DINOv2's 768 numbers in 8 views
(`piece_features`). Per outer round, fitted on the training settings only:

1. z-score and PCA to 32 numbers (`PiecePCA`), fitted on the training pieces' views, then frozen;
2. a shared piece encoder, Linear 32 -> 16 + ReLU;
3. gated attention (Ilse et al. 2018): a trust score per piece, softmax over the clip's pieces;
4. the trust-weighted mean of the piece vectors -> Linear 16 -> 1 = the clip's size (z-scored).

1,106 trainable numbers. Training: AdamW (learning rate 0.01), one step per epoch over all training clips, each clip a
random 64 of its pieces, each piece in a random one of its 8 views; 5 seeds averaged. Weight decay
and epochs are chosen by leaving one *training* setting out (inner folds); the held-out setting is
never seen. Training runs on the CPU in float32 with deterministic algorithms, so the same seeds
give the same numbers.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np
import torch
from torch import nn

from learned_pathway import pathway2

PCA_DIM = 32
HIDDEN = 16
ATTENTION = 16
BAG = 64
LR = 1e-2  # 1e-3 under-trained the SYNTHETIC test (MAE 0.66-0.72 x B0); 1e-2 gives 0.44-0.50
SEEDS = (0, 1, 2, 3, 4)


@dataclass(frozen=True)
class Config:
    weight_decay: float
    epochs: int


GRID = tuple(Config(wd, ep) for wd in (1e-3, 1e-2, 1e-1) for ep in (100, 300))


class PiecePCA:
    """z-score per column (SD < 1e-12 -> 1), then the first PCA_DIM principal directions (SVD)."""

    def fit(self, X: np.ndarray) -> "PiecePCA":
        X = np.asarray(X, np.float64).reshape(-1, np.asarray(X).shape[-1])
        self.mu = X.mean(0)
        self.sd = X.std(0)
        self.sd[self.sd < 1e-12] = 1.0
        _, _, vt = np.linalg.svd((X - self.mu) / self.sd, full_matrices=False)
        self.P = vt[:PCA_DIM].T  # (d, PCA_DIM)
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, np.float64)
        return (((X - self.mu) / self.sd) @ self.P).astype(np.float32)

    def digest_bytes(self) -> bytes:
        return b"".join(np.ascontiguousarray(a, np.float64).tobytes() for a in (self.mu, self.sd, self.P))


class GatedAttentionMIL(nn.Module):
    """Bag (m, PCA_DIM) -> (prediction, weights (m,)); or (B, M, PCA_DIM) with a (B, M) mask."""

    def __init__(self, in_dim: int = PCA_DIM, hidden: int = HIDDEN, attention: int = ATTENTION):
        super().__init__()
        self.encode = nn.Sequential(nn.Linear(in_dim, hidden), nn.ReLU())
        self.V = nn.Linear(hidden, attention)
        self.U = nn.Linear(hidden, attention)
        self.w = nn.Linear(attention, 1)
        self.out = nn.Linear(hidden, 1)

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor]:
        single = x.dim() == 2
        if single:
            x = x[None]
        h = self.encode(x)  # (B, M, hidden)
        score = self.w(torch.tanh(self.V(h)) * torch.sigmoid(self.U(h))).squeeze(-1)  # (B, M)
        if mask is not None:
            score = score.masked_fill(~mask, float("-inf"))
        weights = torch.softmax(score, dim=1)
        y = self.out((weights[..., None] * h).sum(1)).squeeze(-1)  # (B,)
        return (y[0], weights[0]) if single else (y, weights)


def _train(Z: Mapping[str, np.ndarray], yz: Mapping[str, float], config: Config, seed: int,
           curve: list | None = None) -> GatedAttentionMIL:
    """One seed's training on PCA-reduced bags Z[clip] (m, 8, PCA_DIM) and z-scored labels."""
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    net = GatedAttentionMIL()
    opt = torch.optim.AdamW(net.parameters(), lr=LR, weight_decay=config.weight_decay)
    clips = sorted(Z)
    target = torch.tensor([yz[c] for c in clips], dtype=torch.float32)
    size = min(BAG, max(len(Z[c]) for c in clips))
    for _ in range(config.epochs):
        x = torch.zeros(len(clips), size, PCA_DIM)
        mask = torch.zeros(len(clips), size, dtype=torch.bool)
        for b, c in enumerate(clips):
            m = len(Z[c])
            k = min(size, m)
            pick = rng.choice(m, size=k, replace=False)
            view = rng.integers(0, Z[c].shape[1], size=k)
            x[b, :k] = torch.from_numpy(Z[c][pick, view])
            mask[b, :k] = True
        pred, _ = net(x, mask)
        loss = ((pred - target) ** 2).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
        if curve is not None:
            curve.append(loss.item())
    return net


def _prepare(train: Mapping[str, np.ndarray], y: Mapping[str, float]):
    pca = PiecePCA().fit(np.concatenate([np.asarray(v).reshape(-1, np.asarray(v).shape[-1]) for v in train.values()]))
    labels = np.array([y[c] for c in train], float)
    mu, sd = float(labels.mean()), float(labels.std())
    sd = sd if sd > 1e-12 else 1.0
    Z = {c: pca.transform(v) for c, v in train.items()}
    yz = {c: (y[c] - mu) / sd for c in train}
    return pca, mu, sd, Z, yz


def fit_predict(train: Mapping[str, np.ndarray], y: Mapping[str, float], test: Mapping[str, np.ndarray],
                config: Config, seeds: Sequence[int] = SEEDS) -> tuple[dict[str, float], dict[str, np.ndarray], str]:
    """Fit on `train` bags only; predict every `test` bag from all its pieces, view 0.

    Returns (predictions in label units, mean over seeds; attention per test clip, mean over seeds;
    SHA-256 of the PCA and every seed's final weights)."""
    torch.use_deterministic_algorithms(True)
    pca, mu, sd, Z, yz = _prepare(train, y)
    digest = hashlib.sha256(pca.digest_bytes())
    preds = {c: [] for c in test}
    atts = {c: [] for c in test}
    for seed in seeds:
        net = _train(Z, yz, config, seed)
        for t in net.state_dict().values():
            digest.update(t.detach().cpu().numpy().astype(np.float32).tobytes())
        net.eval()
        with torch.no_grad():
            for c, v in test.items():
                p, w = net(torch.from_numpy(pca.transform(np.asarray(v)[:, 0])))
                preds[c].append(float(p) * sd + mu)
                atts[c].append(w.numpy())
    return ({c: float(np.mean(v)) for c, v in preds.items()},
            {c: np.mean(v, axis=0) for c, v in atts.items()}, digest.hexdigest())


def training_curve(train: Mapping[str, np.ndarray], y: Mapping[str, float], config: Config,
                   seed: int = 0) -> list[float]:
    """The training loss after every step of one seed's training, as in `fit_predict`."""
    torch.use_deterministic_algorithms(True)
    _, _, _, Z, yz = _prepare(train, y)
    curve: list[float] = []
    _train(Z, yz, config, seed, curve)
    return curve


def select_config(bags: Mapping[str, np.ndarray], y: Mapping[str, float], clips: Sequence[str],
                  grid: Sequence[Config] = GRID, seeds: Sequence[int] = SEEDS) -> tuple[Config, list[dict]]:
    """Choose weight decay and epochs by leaving one of `clips`' settings out at a time (pooled MAE).
    Ties within 1e-12 go to fewer epochs, then larger weight decay. One candidate: no fitting."""
    clips = list(clips)
    if len(grid) == 1:
        return grid[0], [{"weight_decay": grid[0].weight_decay, "epochs": grid[0].epochs, "mae": float("nan")}]
    folds = pathway2.setting_folds(clips)
    table = []
    for cfg in grid:
        errors = []
        for f in folds:
            pred, _, _ = fit_predict({c: bags[c] for c in f.train}, y, {c: bags[c] for c in f.test}, cfg, seeds)
            errors += [abs(pred[c] - y[c]) for c in f.test]
        table.append({"weight_decay": cfg.weight_decay, "epochs": cfg.epochs, "mae": float(np.mean(errors))})
    best = min(r["mae"] for r in table)
    pick = min((r for r in table if r["mae"] <= best + 1e-12), key=lambda r: (r["epochs"], -r["weight_decay"]))
    return Config(pick["weight_decay"], pick["epochs"]), table


def cross_predict_dl2(bags: Mapping[str, np.ndarray], y: Mapping[str, float],
                      outer_folds: Sequence[pathway2.Round], grid: Sequence[Config] = GRID,
                      seeds: Sequence[int] = SEEDS) -> tuple[dict[str, float], list[dict], dict[str, np.ndarray]]:
    """Each outer test clip's prediction, with the choice and the fit made on the round's training
    clips only. Records: held_out, weight_decay, epochs, inner_mae, n_train, digest."""
    pred: dict[str, float] = {}
    attention: dict[str, np.ndarray] = {}
    records: list[dict] = []
    for rnd in outer_folds:
        cfg, table = select_config(bags, y, rnd.train, grid, seeds)
        inner = next(r["mae"] for r in table if r["weight_decay"] == cfg.weight_decay and r["epochs"] == cfg.epochs)
        p, a, digest = fit_predict({c: bags[c] for c in rnd.train}, y, {c: bags[c] for c in rnd.test}, cfg, seeds)
        for c in p:
            if c in pred:
                raise ValueError(f"clip {c!r} would be predicted twice")
        pred.update(p)
        attention.update(a)
        records.append({"held_out": rnd.held_out, "weight_decay": cfg.weight_decay, "epochs": cfg.epochs,
                        "inner_mae": inner, "n_train": len(rnd.train), "digest": digest})
    return pred, records, attention

"""DL-1: piece crops at a fixed px/mm, clip features, z-score -> PCA -> ridge.

Each measured piece's mask is redrawn as a 224 x 224 picture at exactly 4 px per mm, centred on
the mask's centroid and upright (only the scale of the bolt ruler is used, not its rotation, as
in extruder's measure.rectify). Clip features are the mean and SD (ddof = 1) of the pieces'
descriptions. The model z-scores its inputs, keeps k PCA components and fits a ridge regression
on the scores; every learned number comes from the training clips of its round only.
Pure numpy/OpenCV; the network lives in learned_pathway.dl1_embed.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

import cv2
import numpy as np

from learned_pathway import pathway2

CROP_PX = 224
CROP_PX_PER_MM = 4.0
KS = (4, 8, 16)
ALPHAS = (0.01, 0.1, 1.0, 10.0, 100.0, 1000.0)


def largest_component(mask: np.ndarray) -> np.ndarray:
    """The largest 8-connected part of a mask, as extruder's measure_grid_mask keeps it (loose specks
    are noise). An empty mask stays empty."""
    m = np.asarray(mask, dtype=bool)
    n, lab, stats, _ = cv2.connectedComponentsWithStats(m.astype(np.uint8), connectivity=8)
    if n <= 1:
        return m.copy()
    return lab == 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))


def crop_affine(mask: np.ndarray, px_to_mm: np.ndarray,
                px_per_mm_out: float = CROP_PX_PER_MM, size: int = CROP_PX) -> np.ndarray:
    """The 2 x 3 affine that redraws a crop-pixel mask at `px_per_mm_out`, centred on its centroid.

    Only the stretch of `px_to_mm` (full-frame px -> mm, 2 x 3) is used, so the piece keeps the
    orientation it has in the camera picture. The centroid of the True pixels lands on
    ((size - 1) / 2, (size - 1) / 2); the crop origin therefore never enters.
    """
    m = np.asarray(mask, dtype=bool)
    if m.ndim != 2:
        raise ValueError(f"expected a 2-D mask, got shape {m.shape}")
    ys, xs = np.nonzero(m)
    if len(xs) == 0:
        raise ValueError("crop_affine needs a non-empty mask")
    _, s, vt = np.linalg.svd(np.asarray(px_to_mm, float)[:, :2])
    stretch = vt.T @ np.diag(s) @ vt  # mm per px with the rotation removed
    lin = px_per_mm_out * stretch
    centre = np.array([xs.mean(), ys.mean()])
    trans = (size - 1) / 2.0 - lin @ centre
    return np.hstack([lin, trans[:, None]])


def make_crops(mask: np.ndarray, A: np.ndarray, frame: np.ndarray | None = None,
               size: int = CROP_PX) -> tuple[np.ndarray, np.ndarray | None, bool]:
    """(outline, colour, touches_border): the mask alone, or also the warped frame inside it.

    `frame` is an RGB uint8 (H, W, 3) picture in the same pixel coordinates as the mask; the
    colour crop is zero wherever the outline is. `touches_border` means the piece reaches the
    crop's edge: longer than 56 mm at 4 px/mm, or badly off-centre.
    """
    A = np.asarray(A, np.float64)
    warped = cv2.warpAffine(np.asarray(mask, dtype=np.float32), A, (size, size),
                            flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    outline = np.where(warped >= 0.5, np.uint8(255), np.uint8(0))
    colour: np.ndarray | None = None
    if frame is not None:
        colour = cv2.warpAffine(np.asarray(frame), A, (size, size),
                                flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        colour = colour.copy()
        colour[outline == 0] = 0
    touches = bool(outline[0].any() or outline[-1].any()
                   or outline[:, 0].any() or outline[:, -1].any())
    return outline, colour, touches


def clip_features(features: np.ndarray, clips: Sequence[str]) -> tuple[list[str], np.ndarray]:
    """Per clip, the column means then the column SDs (ddof = 1) of the pieces' features.

    Returns the sorted clip ids and an (n_clips, 2 * d) array. A clip with fewer than 2 pieces
    has no SD, so it is refused.
    """
    f = np.asarray(features, float)
    ids = list(clips)
    if f.ndim != 2 or len(ids) != len(f):
        raise ValueError(f"expected one feature row per clip entry, got {f.shape} and {len(ids)} clips")
    rows, order = [], []
    for cid in sorted(set(ids)):
        group = f[[i for i, c in enumerate(ids) if c == cid]]
        if len(group) < 2:
            raise ValueError(f"clip {cid!r} has fewer than 2 pieces, so no SD can be taken")
        rows.append(np.concatenate([group.mean(0), group.std(0, ddof=1)]))
        order.append(cid)
    return order, np.vstack(rows)


@dataclass
class ZPCARidge:
    """z-score -> PCA (k components, no whitening) -> ridge on the scores, target z-scored.

    There is no intercept: the PCA scores and the z-scored target are centred, so it cancels.
    """
    k: int
    alpha: float

    def fit(self, X: np.ndarray, y: np.ndarray) -> "ZPCARidge":
        X = np.asarray(X, float)
        y = np.asarray(y, float).ravel()
        n, d = X.shape
        if not 1 <= self.k <= min(n - 1, d):
            raise ValueError(f"k = {self.k} is impossible for {n} rows and {d} features")
        self.mu_x = X.mean(0)
        self.sd_x = X.std(0)
        self.sd_x[self.sd_x < 1e-12] = 1.0
        Z = (X - self.mu_x) / self.sd_x
        _, _, vt = np.linalg.svd(Z, full_matrices=False)
        self.P = vt[:self.k].T
        T = Z @ self.P
        self.mu_y = float(y.mean())
        self.sd_y = float(y.std())
        if self.sd_y < 1e-12:
            self.sd_y = 1.0
        yz = (y - self.mu_y) / self.sd_y
        self.w = np.linalg.solve(T.T @ T + self.alpha * np.eye(self.k), T.T @ yz)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        Z = (np.asarray(X, float) - self.mu_x) / self.sd_x
        return (Z @ self.P @ self.w) * self.sd_y + self.mu_y


def inner_folds(clips: Sequence[str], scheme: str) -> list[pathway2.Round]:
    """The inner folds used to choose k and alpha: leave one training setting out, or one clip."""
    if scheme == "setting":
        return pathway2.setting_folds(clips)
    if scheme == "clip":
        return pathway2.clip_folds(clips)
    raise ValueError(f"unknown inner scheme {scheme!r}; expected 'setting' or 'clip'")


def select_k_alpha(F: Mapping[str, np.ndarray], y: Mapping[str, float], clips: Sequence[str],
                   scheme: str, ks: Sequence[int] = KS,
                   alphas: Sequence[float] = ALPHAS) -> tuple[int, float, list[dict]]:
    """Choose (k, alpha) by pooled inner MAE in mm, using only `clips` (the outer training clips).

    A k is a candidate only if every inner fold can fit it (k <= inner training clips - 1, and
    k <= the feature count). Ties within 1e-12 go to the smaller k, then the larger alpha.
    Returns (k, alpha, table) with {"k", "alpha", "mae"} for every candidate.
    """
    clips = list(clips)
    if not clips:
        raise ValueError("select_k_alpha needs at least one training clip")
    folds = inner_folds(clips, scheme)
    d = len(np.asarray(F[clips[0]], float).ravel())

    def X_of(cs: Sequence[str]) -> np.ndarray:
        return np.vstack([np.asarray(F[c], float).ravel() for c in cs])

    def y_of(cs: Sequence[str]) -> np.ndarray:
        return np.array([y[c] for c in cs], float)

    table: list[dict] = []
    for k in ks:
        if any(k > len(f.train) - 1 or k > d for f in folds):
            continue
        for alpha in alphas:
            errors: list[np.ndarray] = []
            for f in folds:
                pred = ZPCARidge(k, alpha).fit(X_of(f.train), y_of(f.train)).predict(X_of(f.test))
                errors.append(np.abs(pred - y_of(f.test)))
            table.append({"k": int(k), "alpha": float(alpha),
                          "mae": float(np.mean(np.concatenate(errors)))})
    if not table:
        raise ValueError(f"no (k, alpha) candidate is feasible for scheme {scheme!r}")
    best = min(c["mae"] for c in table)
    chosen = min((c for c in table if c["mae"] <= best + 1e-12), key=lambda c: (c["k"], -c["alpha"]))
    return chosen["k"], chosen["alpha"], table


def cross_predict_dl1(F: Mapping[str, np.ndarray], y: Mapping[str, float],
                      outer_folds: Sequence[pathway2.Round], inner_scheme: str,
                      ks: Sequence[int] = KS,
                      alphas: Sequence[float] = ALPHAS) -> tuple[dict[str, float], list[dict]]:
    """Each outer test clip's prediction: k and alpha chosen and the model fitted on the round's
    training clips only. Returns the predictions and one record per round
    ({"held_out", "k", "alpha", "inner_mae", "n_train"})."""
    pred: dict[str, float] = {}
    records: list[dict] = []

    def X_of(cs: Sequence[str]) -> np.ndarray:
        return np.vstack([np.asarray(F[c], float).ravel() for c in cs])

    def y_of(cs: Sequence[str]) -> np.ndarray:
        return np.array([y[c] for c in cs], float)

    for rnd in outer_folds:
        k, alpha, table = select_k_alpha(F, y, rnd.train, inner_scheme, ks, alphas)
        inner_mae = next(c["mae"] for c in table if c["k"] == k and c["alpha"] == alpha)
        model = ZPCARidge(k, alpha).fit(X_of(rnd.train), y_of(rnd.train))
        for clip, value in zip(rnd.test, model.predict(X_of(rnd.test))):
            if clip in pred:
                raise ValueError(f"clip {clip!r} would be predicted twice")
            pred[clip] = float(value)
        records.append({"held_out": rnd.held_out, "k": k, "alpha": alpha,
                        "inner_mae": inner_mae, "n_train": len(rnd.train)})
    return pred, records

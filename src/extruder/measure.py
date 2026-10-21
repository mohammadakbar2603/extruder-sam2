"""Size of one piece from its mask, in millimetres (or pixels if there is no calibration).

Plain words
-----------
1. The mask (crop pixels) is redrawn on a millimetre grid using the die-plate mapping
   (20 grid cells per mm), so every measurement below is in real units.
2. We find the piece's *middle line* (its skeleton) and follow the longest path along it.
3. From that we measure:

   area_mm2        how much of the picture the piece covers
   length_mm       along the middle line, from tip to tip
   width_mm        typical thickness: 2 x distance from the middle line to the edge,
                   median over the central 60 % of the piece (ends are rounded, so skipped)
   width_max_mm    thickest point
   width_mean_mm   area / length (an average thickness)
   feret_mm        longest straight distance across the piece (like a caliper's maximum)
   straightness    tip-to-tip distance / length (1 = straight, smaller = more curved)
   expansion       (width_mm / die hole diameter)^2, the usual "sectional expansion ratio"
                   (apparent: seen from one side, hot, right at the die)
"""
import cv2
import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import dijkstra
from skimage.morphology import skeletonize

from .calib import DIE_HOLE_DIAMETER_MM

GRID_PER_MM = 20


def rectify(mask, crop_origin, px_to_mm, grid_per_mm=GRID_PER_MM):
    """Redraw a crop mask on a millimetre grid. Returns (grid mask, cells per unit)."""
    ys, xs = np.nonzero(mask)
    if len(xs) == 0:
        return None, grid_per_mm
    ox, oy = crop_origin
    # crop px -> frame px -> mm -> grid cells
    A = px_to_mm.copy()
    A[:, 2] = A[:, 2] + A[:, :2] @ np.array([ox, oy], float)
    pts = np.array([[xs.min(), ys.min()], [xs.max(), ys.min()], [xs.min(), ys.max()], [xs.max(), ys.max()]], float)
    mm = pts @ A[:, :2].T + A[:, 2]
    lo = mm.min(0) - 1.0
    hi = mm.max(0) + 1.0
    G = A * grid_per_mm
    G[:, 2] -= lo * grid_per_mm
    size = np.ceil((hi - lo) * grid_per_mm).astype(int)
    out = cv2.warpAffine(mask.astype(np.uint8), G, (int(size[0]), int(size[1])), flags=cv2.INTER_LINEAR)
    return out > 0, grid_per_mm


def _longest_path(skel):
    """Pixels of the longest path through a skeleton (list of (y, x)) and its length in cells."""
    ys, xs = np.nonzero(skel)
    n = len(ys)
    if n < 2:
        return np.stack([ys, xs], 1), 0.0
    idx = -np.ones(skel.shape, int)
    idx[ys, xs] = np.arange(n)
    rows, cols, w = [], [], []
    for dy, dx in ((0, 1), (1, 0), (1, 1), (1, -1)):
        y2, x2 = ys + dy, xs + dx
        ok = (y2 >= 0) & (y2 < skel.shape[0]) & (x2 >= 0) & (x2 < skel.shape[1])
        j = np.full(n, -1)
        j[ok] = idx[y2[ok], x2[ok]]
        ok = j >= 0
        rows += list(np.arange(n)[ok]); cols += list(j[ok]); w += [np.hypot(dy, dx)] * int(ok.sum())
    g = coo_matrix((w + w, (rows + cols, cols + rows)), shape=(n, n)).tocsr()
    d0 = dijkstra(g, indices=0)
    a = int(np.nanargmax(np.where(np.isinf(d0), -1, d0)))
    da, pred = dijkstra(g, indices=a, return_predecessors=True)
    b = int(np.nanargmax(np.where(np.isinf(da), -1, da)))
    path = [b]
    while path[-1] != a and pred[path[-1]] >= 0:
        path.append(int(pred[path[-1]]))
    return np.stack([ys[path], xs[path]], 1), float(da[b])


def measure_grid_mask(m, cells_per_unit, unit="mm", die_hole=DIE_HOLE_DIAMETER_MM):
    """All measurements of one mask that is already on a regular grid."""
    out = {}
    m = m.astype(bool)
    if m.sum() < 10:
        return out
    k = cells_per_unit
    # keep the largest connected part (tiny specks are noise)
    n, lab, stats, _ = cv2.connectedComponentsWithStats(m.astype(np.uint8), connectivity=8)
    m = lab == (1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA])))
    dt = cv2.distanceTransform(m.astype(np.uint8), cv2.DIST_L2, 5) / k
    path, plen = _longest_path(skeletonize(m))
    r = dt[path[:, 0], path[:, 1]] if len(path) else np.array([dt.max()])
    lo, hi = int(0.2 * len(r)), max(int(0.8 * len(r)), int(0.2 * len(r)) + 1)
    area = m.sum() / k ** 2
    length = plen / k + (r[0] + r[-1] if len(r) > 1 else 2 * r[0])
    width = 2 * float(np.median(r[lo:hi]))
    hull = cv2.convexHull(np.stack(np.nonzero(m)[::-1], 1).astype(np.int32))[:, 0]
    dd = np.hypot(*(hull[:, None, :] - hull[None, :, :]).transpose(2, 0, 1))
    tip = np.hypot(*(path[0] - path[-1])) / k if len(path) > 1 else 0.0
    out[f"area_{unit}2"] = float(area)
    out[f"length_{unit}"] = float(length)
    out[f"width_{unit}"] = width
    out[f"width_max_{unit}"] = float(2 * r.max())
    out[f"width_mean_{unit}"] = float(area / length) if length > 0 else np.nan
    out[f"feret_{unit}"] = float(dd.max() / k)
    out["straightness"] = float(min(1.0, (tip + (r[0] + r[-1] if len(r) > 1 else 0)) / length)) if length > 0 else np.nan
    if unit == "mm":
        out["expansion"] = float((width / die_hole) ** 2)
    return out


def measure_mask(mask, crop_origin=(0, 0), px_to_mm=None):
    """Measure one crop mask. Always gives pixel values; adds mm values when calibrated."""
    res = {"area_px": int(mask.sum())}
    px = measure_grid_mask(mask, 1.0, unit="px")
    res.update({k: v for k, v in px.items() if k not in ("straightness", "area_px2")})
    if px_to_mm is not None:
        g, k = rectify(mask, crop_origin, px_to_mm)
        if g is not None:
            res.update(measure_grid_mask(g, k, unit="mm"))
    else:
        res["straightness"] = px.get("straightness", np.nan)
    return res

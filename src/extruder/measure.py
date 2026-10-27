"""Size of one piece from its mask, in millimetres (or pixels if there is no calibration).

Plain words
-----------
1. The mask (crop pixels) is redrawn on a millimetre grid using the die-plate mapping
   (20 grid cells per mm), so every measurement below is in real units.
2. Along the piece we place the biggest circles that fit inside it; their centres form the
   middle line (see `centerline`).
3. LENGTH is measured like the lab does it with a thread laid along the piece: the length
   of the line through the circle centres, from tip to tip, following the bend.
4. DIAMETER: each circle's diameter is the local thickness. The diameter is the MEDIAN
   thickness over the central 60 % of the piece (the rounded or cut ends are skipped).
5. Every piece is measured at 4 rotations and the median is kept, so the result does not
   depend on which way the piece points (see `measure_rotations`).

   length_mm             tip to tip along the bent middle line (the lab's thread length)
   width_mm              typical thickness (the diameter), as described above
   width_max_mm          thickest point
   width_mean_mm         area / length (an average thickness)
   feret_mm              longest straight distance across the piece (reference only)
   area_mm2              how much of the picture the piece covers
   straightness          straight distance / length along the bend (1 = straight)
   expansion             (width_mm / die hole diameter)^2, kept for reference; the team
                         computes the expansion ratio itself from the diameters
   shape_stability_pct   how much width/length change between the 4 rotations (> 5 % = odd mask)
"""
import cv2
import numpy as np
from scipy.ndimage import binary_fill_holes as ndi_fill, gaussian_filter1d
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
    # crop px -> frame px -> mm -> grid cells. Only the scale/stretch part of the mapping is
    # used, not its rotation, so the piece keeps the orientation it has in the camera picture
    # (sizes do not depend on rotation).
    U, S, Vt = np.linalg.svd(px_to_mm[:, :2])
    stretch = Vt.T @ np.diag(S) @ Vt
    A = np.hstack([stretch, (stretch @ np.array([ox, oy], float))[:, None]])
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


FULL_CIRCLE = 0.95   # a circle counts as "full size" if its radius is >= 95 % of the typical radius
DIR_FROM, DIR_TO = 0.0, 1.0   # end direction measured between these distances (in radii) from the end
SNAP = 10            # grid cells (0.5 mm): read each circle size from the distance ridge this close to the line


def centerline(m, cells_per_unit, full=None):
    """The piece's middle line through the centres of the biggest circles that fit inside.

    1. distance map: every pixel inside gets its distance to the nearest edge. On the middle
       line, this is the radius of the biggest circle centred there that fits in the piece;
    2. skeleton: the (slightly smoothed) shape thinned to a 1-pixel middle line; keep its
       longest path = the chain of circle centres from one end to the other;
    3. smooth that chain (a path made of pixel steps is longer than the real line);
    4. near each end the circles shrink (rounded tip) or the chain forks into the corners
       (flat, knife-cut face). We keep the chain only between the first and the last
       *full-size* circle (radius >= 95 % of the typical radius);
    5. from the last full-size circle at each end, go straight on (about one radius, in the
       direction of the chain) until the edge of the piece: that is the tip.
       LENGTH = chain of centres + the two short end pieces.
    Returns a dict (lengths in grid cells): dt, path (kept chain of centres), r (radius along
    it), centres (where each of those circles sits), tips, ext (the two end pieces), length,
    r_med, full_path (raw skeleton path).
    """
    full = FULL_CIRCLE if full is None else full
    k = cells_per_unit
    dt = cv2.distanceTransform(m.astype(np.uint8), cv2.DIST_L2, cv2.DIST_MASK_PRECISE)
    # skeleton of a slightly smoothed, hole-free outline: ragged edges would add side branches
    sm = cv2.GaussianBlur(ndi_fill(m).astype(np.float32), (0, 0), max(1.0, 0.4 * k)) > 0.5
    sk = skeletonize(sm)
    n_sk, lab_sk = cv2.connectedComponents(sk.astype(np.uint8), connectivity=8)
    if n_sk > 2:  # keep the biggest skeleton piece
        sizes = np.bincount(lab_sk.ravel())[1:]
        sk = lab_sk == 1 + int(np.argmax(sizes))
    path, _ = _longest_path(sk)
    if len(path) < 5:
        rr = float(dt.max())
        return dict(dt=dt, path=path.astype(float), r=np.array([rr]), tips=[], ext=(rr, rr), length=2 * rr,
                    r_med=rr, full_path=path)
    # circle size at each centre: the distance-map ridge right next to the line (the smoothed
    # outline can shift the thin line slightly off the true centre on bends)
    centres = path.copy()
    if SNAP:
        disk = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * SNAP + 1,) * 2) > 0
        H, W = dt.shape
        for i, (y, x) in enumerate(path):
            y0, x0 = max(0, y - SNAP), max(0, x - SNAP)
            win = dt[y0:y + SNAP + 1, x0:x + SNAP + 1] * disk[SNAP - (y - y0):SNAP + (min(H, y + SNAP + 1) - y),
                                                               SNAP - (x - x0):SNAP + (min(W, x + SNAP + 1) - x)]
            dy, dx = np.unravel_index(int(np.argmax(win)), win.shape)
            centres[i] = (y0 + dy, x0 + dx)
    r_all = dt[centres[:, 0], centres[:, 1]]
    r_raw = dt[path[:, 0], path[:, 1]]   # used to decide where the full-size circles stop
    n = len(r_all)
    central = slice(int(0.2 * n), max(int(0.8 * n), int(0.2 * n) + 1))
    r_med = float(np.median(r_all[central]))
    big = np.nonzero(r_raw >= full * float(np.median(r_raw[central])))[0]
    i0, i1 = (int(big[0]), int(big[-1])) if len(big) >= 2 else (0, n - 1)
    sm_path = gaussian_filter1d(path.astype(float), max(2.0, 0.5 * k), axis=0, mode="nearest")
    mid = sm_path[i0:i1 + 1]
    r = r_all[i0:i1 + 1]
    chain = float(np.sum(np.hypot(*np.diff(mid, axis=0).T)))
    # from the last full-size circle, go straight on (about one radius) until the edge of the piece
    a2 = np.concatenate([[0], np.cumsum(np.hypot(*np.diff(mid, axis=0).T))])
    tips, ext = [], []
    for end in (0, -1):
        p = mid[end]
        from_end = a2 if end == 0 else a2[-1] - a2
        # direction of the chain a little way in from the end (the very end can lean into a corner)
        j_near = int(np.argmin(np.abs(from_end - DIR_FROM * r_med)))
        j_far = int(np.argmin(np.abs(from_end - DIR_TO * r_med)))
        v = mid[j_near] - mid[j_far]
        nv = np.linalg.norm(v)
        t = 0.0
        if nv > 1e-6:
            v = v / nv
            for _ in range(4 * sum(m.shape)):
                q = p + v * (t + 0.25)
                yi, xi = int(round(q[0])), int(round(q[1]))
                if not (0 <= yi < m.shape[0] and 0 <= xi < m.shape[1]) or not m[yi, xi]:
                    break
                t += 0.25
        else:
            t, v = float(r[end]), np.zeros(2)
        tips.append(p + v * t)
        ext.append(t)
    ext = tuple(ext)
    return dict(dt=dt, path=mid, r=r, centres=centres[i0:i1 + 1], tips=tips, ext=ext,
                length=chain + ext[0] + ext[1], r_med=r_med, full_path=path)


def measure_grid_mask(m, cells_per_unit, unit="mm", die_hole=DIE_HOLE_DIAMETER_MM):
    """All measurements of one mask that is already on a regular grid."""
    out = {}
    m = m.astype(bool)
    if m.sum() < 10:
        return out
    k = cells_per_unit
    # empty margin all round: the distance map must see the piece's edge, not the picture border
    m = np.pad(m, 3)
    # keep the largest connected part (tiny specks are noise)
    n, lab, stats, _ = cv2.connectedComponentsWithStats(m.astype(np.uint8), connectivity=8)
    m = lab == (1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA])))
    c = centerline(m, k)
    r = c["r"] / k
    n = len(r)
    lo, hi = int(0.2 * n), max(int(0.8 * n), int(0.2 * n) + 1)
    area = m.sum() / k ** 2
    curve = c["length"] / k
    width = 2 * float(np.median(r[lo:hi]))
    hull = cv2.convexHull(np.stack(np.nonzero(m)[::-1], 1).astype(np.int32))[:, 0]
    dd = np.hypot(*(hull[:, None, :] - hull[None, :, :]).transpose(2, 0, 1))
    straight = float(dd.max() / k)   # longest straight distance across the piece
    out[f"area_{unit}2"] = float(area)
    out[f"length_{unit}"] = float(curve)   # along the bend, like the lab's thread
    out[f"feret_{unit}"] = straight
    out[f"width_{unit}"] = width
    out[f"width_max_{unit}"] = float(2 * r.max())
    out[f"width_mean_{unit}"] = float(area / curve) if curve > 0 else np.nan
    out["straightness"] = float(min(1.0, straight / curve)) if curve > 0 else np.nan
    if unit == "mm":
        out["expansion"] = float((width / die_hole) ** 2)
    return out


ROTATIONS = (0, 45, 90, 135)   # degrees


def rotate_grid_mask(m, angle):
    """Turn a grid mask by `angle` degrees (no resizing) and crop it tightly."""
    g = np.pad(m.astype(np.uint8), max(m.shape) // 2 + 5)
    c = (g.shape[1] / 2, g.shape[0] / 2)
    g = cv2.warpAffine(g, cv2.getRotationMatrix2D(c, angle, 1.0), g.shape[::-1], flags=cv2.INTER_NEAREST) > 0
    ys, xs = np.nonzero(g)
    return g[ys.min():ys.max() + 1, xs.min():xs.max() + 1] if len(ys) else g


def measure_rotations(m, cells_per_unit, unit="mm", angles=ROTATIONS):
    """Measure the same piece turned to several angles and keep the median of each value.

    A good mask gives almost the same numbers at every angle (typically within 1-2 %).
    `shape_stability_pct` is the larger of the width and length variation between angles;
    a high value (> 5 %) means a ragged or odd-shaped mask (motion blur, two pieces merged),
    which is better left out of the clip average.
    """
    outs = [measure_grid_mask(rotate_grid_mask(m, a), cells_per_unit, unit) for a in angles]
    outs = [o for o in outs if o]
    if not outs:
        return {}
    res = {key: float(np.median([o[key] for o in outs])) for key in outs[0]}
    cv = lambda key: float(np.std([o[key] for o in outs]) / (np.mean([o[key] for o in outs]) + 1e-9) * 100)
    res["shape_stability_pct"] = max(cv(f"width_{unit}"), cv(f"length_{unit}"))
    return res


def measure_mask(mask, crop_origin=(0, 0), px_to_mm=None):
    """Measure one crop mask. Always gives pixel values; adds mm values when calibrated.

    The mm values are the median over 4 rotations of the piece (see `measure_rotations`),
    so the result does not depend on which way the piece happens to point.
    """
    res = {"area_px": int(mask.sum())}
    px = measure_grid_mask(mask, 1.0, unit="px")
    res.update({k: v for k, v in px.items() if k not in ("straightness", "area_px2")})
    if px_to_mm is not None:
        g, k = rectify(mask, crop_origin, px_to_mm)
        if g is not None:
            res.update(measure_rotations(g, k, unit="mm"))
    else:
        res["straightness"] = px.get("straightness", np.nan)
    return res

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
from skimage.graph import MCP_Geometric, route_through_array

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


FULL_CIRCLE = 0.85   # a circle is "full size" at >= 85 % of the typical radius
TIP_ZONE = 2.0       # look for the last full-size circle only within this many radii of each tip
SNAP = 10            # grid cells (0.5 mm): read each circle size from the distance ridge this close to the line
RIDGE_POWER = 8.0    # how strongly the middle line is pulled onto the ridge of biggest circles


def _farthest_inside(cost, start):
    """Point inside the piece that is farthest from `start` when walking only inside it."""
    d, _ = MCP_Geometric(cost).find_costs([tuple(start)])
    d[~np.isfinite(d)] = -1
    return np.unravel_index(int(np.argmax(d)), d.shape)


def centerline(m, cells_per_unit, step=2):
    """Middle line that follows the piece like a thread laid along it, from tip to tip.

    1. distance map: every inside point gets its distance to the nearest edge (on the middle
       line = radius of the biggest circle that fits there);
    2. the two TIPS: start anywhere inside, walk (only inside the piece) to the farthest point:
       that is one tip; from there, the farthest point again is the other tip;
    3. the path between the two tips that stays on the ridge of biggest circles (walking is
       cheap where the circles are big, expensive near the edge), so it follows bends and curls
       like a thread would;
    4. smooth it (a path made of pixel steps is longer than the real line);
    5. the two ends: near each tip, the last full-size circle (>= 85 % of the typical radius)
       touches the tip (rounded end) or the cut face (flat end), so the end is counted as one
       radius beyond it. Without this, a path ending in a corner of a flat cut face would be
       about 40 % of a radius too long at that end.
       LENGTH = path between those two circles + one radius at each end.
    The search runs on a 2x coarser grid (10 cells per mm) for speed; distances use the full grid.
    Returns a dict (lengths in grid cells): dt, path (smoothed tip-to-tip path), r (radius of the
    biggest circle along it), centres (where each circle sits), tips, length, r_med, full_path.
    """
    k = cells_per_unit
    dt = cv2.distanceTransform(m.astype(np.uint8), cv2.DIST_L2, cv2.DIST_MASK_PRECISE)
    ms = cv2.resize(m.astype(np.uint8), (m.shape[1] // step, m.shape[0] // step), interpolation=cv2.INTER_AREA) > 0
    ms = ndi_fill(ms)
    dts = cv2.distanceTransform(ms.astype(np.uint8), cv2.DIST_L2, cv2.DIST_MASK_PRECISE)
    if dts.max() <= 0:
        rr = float(dt.max())
        return dict(dt=dt, path=np.zeros((0, 2)), r=np.array([rr]), centres=np.zeros((0, 2), int), tips=[],
                    length=2 * rr, r_med=rr, full_path=np.zeros((0, 2), int))
    walk = np.where(ms, 1.0, np.inf)
    seed = np.unravel_index(int(np.argmax(dts)), dts.shape)
    a = _farthest_inside(walk, seed)
    b = _farthest_inside(walk, a)
    ridge_cost = np.where(ms, 1.0 / (dts / dts.max() + 0.02) ** RIDGE_POWER, np.inf)
    coarse, _ = route_through_array(ridge_cost, a, b, fully_connected=True, geometric=True)
    path = np.clip(np.array(coarse) * step + step // 2, 0, np.array(m.shape) - 1)
    full_path = path.copy()
    sm_path = gaussian_filter1d(path.astype(float), max(2.0, 0.5 * k), axis=0, mode="nearest")
    # circle size at each point: the distance-map ridge right next to the path
    centres = np.clip(np.round(sm_path).astype(int), 0, np.array(m.shape) - 1)
    if SNAP:
        disk = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * SNAP + 1,) * 2) > 0
        H, W = dt.shape
        for i, (y, x) in enumerate(centres.copy()):
            y0, x0 = max(0, y - SNAP), max(0, x - SNAP)
            win = dt[y0:y + SNAP + 1, x0:x + SNAP + 1] * disk[SNAP - (y - y0):SNAP + (min(H, y + SNAP + 1) - y),
                                                               SNAP - (x - x0):SNAP + (min(W, x + SNAP + 1) - x)]
            dy, dx = np.unravel_index(int(np.argmax(win)), win.shape)
            centres[i] = (y0 + dy, x0 + dx)
    r = dt[centres[:, 0], centres[:, 1]]
    n = len(r)
    r_med = float(np.median(r[int(0.2 * n):max(int(0.8 * n), int(0.2 * n) + 1)]))
    # the tips: at a flat (knife-cut) face the path ends in a corner of the face, which is
    # about 40 % of a radius too far. Within the last few radii of each end, find the last
    # full-size circle; the tip or the cut face is one radius beyond it. The search is limited
    # to the ends, so a curl further in cannot stop it early.
    seg = np.hypot(*np.diff(sm_path, axis=0).T)
    arc = np.concatenate([[0], np.cumsum(seg)])
    yi, xi = np.clip(np.round(sm_path).astype(int), 0, np.array(m.shape) - 1).T
    r_on = dt[yi, xi]
    i0, i1, e0, e1 = 0, n - 1, 0.0, 0.0
    head = np.nonzero((arc <= TIP_ZONE * r_med) & (r_on >= FULL_CIRCLE * r_med))[0]
    tail = np.nonzero((arc >= arc[-1] - TIP_ZONE * r_med) & (r_on >= FULL_CIRCLE * r_med))[0]
    if len(head) and len(tail) and head[0] < tail[-1]:
        i0, i1 = int(head[0]), int(tail[-1])
        e0, e1 = float(r_on[i0]), float(r_on[i1])
    length = float(arc[i1] - arc[i0]) + e0 + e1
    return dict(dt=dt, path=sm_path, r=r, centres=centres, tips=[sm_path[0], sm_path[-1]],
                length=length, r_med=r_med, full_path=full_path, ends=(i0, i1), end_radius=(e0, e1))


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


def outline_roughness(m, cells_per_mm, smooth_mm=0.5):
    """How ragged the outline is, in % of the piece's area.

    The mask is compared with a smoothed copy of itself (outline blurred over ~0.5 mm, holes
    filled). A clean, smooth mask hardly changes (about 1-2 %); a speckled, torn or holey mask
    (motion blur, steam, two pieces touching) changes a lot.
    """
    m = np.pad(m.astype(bool), int(3 * smooth_mm * cells_per_mm) + 3)
    sm = cv2.GaussianBlur(ndi_fill(m).astype(np.float32), (0, 0), smooth_mm * cells_per_mm) > 0.5
    return float(np.logical_xor(m, sm).sum() / max(1, sm.sum()) * 100)


# ---- dropping bad masks (thresholds set by eye on clip 0069; to be confirmed on hand-labelled masks)
MAX_ROUGHNESS_PCT = 2.5       # ragged outline: speckles, holes, torn edges
TUBE_FILL_RANGE = (0.80, 1.20)  # area / (length x diameter); single tubes measure 0.88-1.03, merges 1.22+
MAX_STABILITY_PCT = 5.0       # measurement changes too much when the piece is turned


def mask_problem(res):
    """Why a measured mask should be left out of the clip average ("" = fine)."""
    if res.get("roughness_pct", 0) > MAX_ROUGHNESS_PCT:
        return "ragged_outline"
    fill = res.get("tube_fill")
    if fill is not None and fill > TUBE_FILL_RANGE[1]:
        return "extra_area_merged_piece"
    if fill is not None and fill < TUBE_FILL_RANGE[0]:
        return "missing_area_cut_off_piece"
    if res.get("shape_stability_pct", 0) > MAX_STABILITY_PCT:
        return "unstable_measurement"
    return ""


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
            res["roughness_pct"] = outline_roughness(g, k)
            if res.get("length_mm") and res.get("width_mm"):
                res["tube_fill"] = res["area_mm2"] / (res["length_mm"] * res["width_mm"])
            res["mask_problem"] = mask_problem(res)
    else:
        res["straightness"] = px.get("straightness", np.nan)
    return res

"""Explanation figures: how one piece is measured, and how bad masks are dropped.

Same figures as reports/how_we_measure_a_puff.pdf and reports/how_we_drop_bad_masks.pdf,
for any video and any cut.
"""
import json
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Circle
from PIL import Image
from scipy.ndimage import binary_fill_holes
from skimage.graph import MCP_Geometric

from . import measure

COL = {"blue": (60, 200, 255), "red": (255, 60, 60)}
K = measure.GRID_PER_MM


class Piece:
    """One RED or BLUE mask of one cut, ready to draw."""

    def __init__(self, out_dir, cut, key, px_to_mm):
        out_dir = Path(out_dir)
        table = pd.read_csv(out_dir / "cuts.csv")
        row = table[table.cut == cut].iloc[0]
        self.cut, self.key = int(cut), key
        self.frame = int(row[f"{key}_frame"])
        self.mask = np.asarray(Image.open(row[f"{key}_mask_file"])) > 0
        self.crop = np.asarray(Image.open(out_dir / "crop_frames" / f"{self.frame:05d}.jpg").convert("RGB"))
        self.origin = tuple(json.loads((out_dir / "crop_frames" / "meta.json").read_text())["origin"])
        self.px_to_mm = px_to_mm
        g, _ = measure.rectify(self.mask, self.origin, px_to_mm)
        g = np.pad(g, 3)
        n, lab, st, _ = cv2.connectedComponentsWithStats(g.astype(np.uint8), connectivity=8)
        self.m = lab == 1 + int(np.argmax(st[1:, cv2.CC_STAT_AREA]))
        self.video = out_dir.name


def _frame_panel(ax, p, title):
    ov = p.crop.astype(np.float32).copy()
    ov[p.mask] = 0.45 * ov[p.mask] + 0.55 * np.array(COL[p.key])
    ys, xs = np.nonzero(p.mask)
    cx, cy = xs.mean(), ys.mean()
    ax.imshow(ov.astype(np.uint8)); ax.set_xlim(cx - 110, cx + 110); ax.set_ylim(cy + 110, cy - 110)
    ax.axis("off"); ax.set_title(title, fontsize=10)


def measure_steps(out_dir, cut, key, px_to_mm, path=None):
    """The 8-panel figure: mask -> mm -> tips -> thread -> circles -> length -> diameter."""
    p = Piece(out_dir, cut, key, px_to_mm)
    m = p.m
    c = measure.centerline(m, K)
    final = measure.measure_mask(p.mask, p.origin, px_to_mm)
    path_, r, cen = c["path"], c["r"] / K, c["centres"]
    s = np.concatenate([[0], np.cumsum(np.hypot(*np.diff(path_, axis=0).T))]) / K
    n = len(r); lo, hi = int(0.2 * n), max(int(0.8 * n), int(0.2 * n) + 1)
    dia = 2 * r
    d_med, d_mean, d_max, d_min = np.median(dia[lo:hi]), dia[lo:hi].mean(), dia[lo:hi].max(), dia[lo:hi].min()
    L = c["length"] / K
    H, W = m.shape
    ext = (0, W / K, H / K, 0)
    xy = lambda q: (q[1] / K, q[0] / K)
    ppm = 1 / np.sqrt(abs(np.linalg.det(px_to_mm[:, :2])))
    step = 2
    ms = binary_fill_holes(cv2.resize(m.astype(np.uint8), (W // step, H // step), interpolation=cv2.INTER_AREA) > 0)
    a = np.clip(np.round(path_[0] / step).astype(int), 0, np.array(ms.shape) - 1)
    walk, _ = MCP_Geometric(np.where(ms, 1.0, np.inf)).find_costs([tuple(a)])
    walk = np.where(np.isfinite(walk), walk * step / K, np.nan)

    fig = plt.figure(figsize=(21, 12))
    gs = fig.add_gridspec(2, 4, hspace=0.36, wspace=0.22)
    ax = [fig.add_subplot(gs[i // 4, i % 4]) for i in range(8)]
    _frame_panel(ax[0], p, f"0. Video frame {p.frame} (cut {cut}) with the {key.upper()} mask\n"
                 "pieces point in any direction and are bent")
    ys, xs = np.nonzero(p.mask)
    y0, y1, x0, x1 = ys.min() - 4, ys.max() + 5, xs.min() - 4, xs.max() + 5
    ax[1].imshow(p.mask[y0:y1, x0:x1], cmap="gray", extent=(x0, x1, y1, y0))
    ax[1].set(xlabel="x (camera pixels)", ylabel="y (camera pixels)")
    ax[1].set_title(f"1. The mask alone: {p.mask.sum()} pixels\nscale from the reference points: "
                    f"1 px = {1 / ppm:.3f} mm", fontsize=10)
    ax[2].imshow(m, cmap="gray", extent=ext)
    ax[2].plot([0.6, 5.6], [0.8, 0.8], color="lime", lw=4); ax[2].text(0.6, 2.2, "5 mm", color="lime")
    ax[2].set(xlabel="mm", ylabel="mm")
    ax[2].set_title(f"2. Same shape redrawn in millimetres\n(20 grid cells per mm). Area {m.sum() / K**2:.0f} mm²",
                    fontsize=10)
    im = ax[3].imshow(walk, cmap="magma", extent=ext)
    plt.colorbar(im, ax=ax[3], fraction=0.046, label="walking distance from tip A, inside the piece (mm)")
    for q, lab in ((path_[0], "tip A"), (path_[-1], "tip B")):
        ax[3].plot(*xy(q), "o", color="yellow", mec="k", ms=9)
        ax[3].annotate(lab, xy(q), xytext=(6, 6), textcoords="offset points", color="yellow", fontsize=10,
                       weight="bold")
    ax[3].set_title("3. Find the two tips: walking only inside the\npiece, tip A is the farthest point from the "
                    "middle;\ntip B is the farthest point from tip A", fontsize=10)
    im = ax[4].imshow(np.where(m, c["dt"] / K, np.nan), cmap="viridis", extent=ext)
    ax[4].plot(path_[:, 1] / K, path_[:, 0] / K, "-", color="red", lw=1.6)
    plt.colorbar(im, ax=ax[4], fraction=0.046, label="distance to nearest edge (mm)")
    ax[4].set_title("4. Distance map (bright = deep inside). The path\nfrom tip A to tip B keeps to the bright ridge,\n"
                    "like a thread laid along the middle", fontsize=10)
    ax[5].imshow(m, cmap="gray", extent=ext, alpha=0.35)
    ax[5].plot(path_[:, 1] / K, path_[:, 0] / K, "-", color="red", lw=1.2)
    for i in np.linspace(0, n - 1, 15).astype(int):
        central = lo <= i < hi
        ax[5].add_patch(Circle(xy(cen[i]), r[i], fill=False, lw=1.8 if central else 1.0,
                               ec="lime" if central else "0.5", ls="-" if central else "--"))
        ax[5].plot(*xy(cen[i]), ".", color="k", ms=4)
    ax[5].set_title("5. Along the path: the biggest circle that fits\ninside the piece. green = central 60 % "
                    "(used for\nthe diameter), grey dashed = the ends", fontsize=10)
    i0, i1 = c["ends"]
    e0, e1 = np.array(c["end_radius"]) / K
    ax[6].imshow(m, cmap="gray", extent=ext, alpha=0.35)
    ax[6].plot(path_[:, 1] / K, path_[:, 0] / K, "-", color="0.5", lw=1.2, label="whole path tip to tip")
    ax[6].plot(path_[i0:i1 + 1, 1] / K, path_[i0:i1 + 1, 0] / K, "-", color="red", lw=3,
               label=f"between the end circles {(s[i1] - s[i0]):.1f} mm")
    for i, e in ((i0, e0), (i1, e1)):
        ax[6].add_patch(Circle(xy(path_[i]), e, fill=False, ec="magenta", lw=2))
        ax[6].plot(*xy(path_[i]), "o", color="magenta", ms=5)
    ax[6].plot([], [], "o", mfc="none", mec="magenta", mew=2, label=f"last full-size circles: + {e0:.1f} + {e1:.1f} mm")
    ax[6].legend(loc="lower right", fontsize=7.5)
    ax[6].set_title("6. LENGTH (like the lab's thread) = path between\nthe last full-size circles + their radius at "
                    "each end\n" + f"= {(s[i1] - s[i0]):.1f} + {e0:.1f} + {e1:.1f} = {L:.1f} mm", fontsize=10)
    ax[7].plot(s, dia, color="k", lw=1.4, label="circle diameter along the piece")
    ax[7].axvspan(s[lo], s[hi - 1], color="lime", alpha=0.18, label="central 60 % (used)")
    ax[7].axhline(d_med, color="green", lw=2.5, label=f"MEDIAN = DIAMETER {d_med:.2f} mm")
    ax[7].axhline(d_mean, color="blue", lw=1, ls="--", label=f"mean {d_mean:.2f}")
    ax[7].axhline(d_max, color="orange", lw=1, ls="--", label=f"biggest {d_max:.2f}")
    ax[7].axhline(d_min, color="purple", lw=1, ls="--", label=f"smallest {d_min:.2f}")
    ax[7].set(xlabel="position along the path (mm)", ylabel="circle diameter (mm)", ylim=(0, max(dia) * 1.35))
    ax[7].legend(fontsize=7.5, loc="lower center", ncol=2)
    ax[7].set_title("7. DIAMETER: the circle diameters along the\npiece; the ends taper, so only the central 60 %\n"
                    "counts, and we take their MEDIAN", fontsize=10)
    fig.suptitle(f"How one {key.upper()} piece is measured ({p.video}, cut {cut}, frame {p.frame}).   "
                 f"This view: diameter {d_med:.2f} mm, length {L:.2f} mm.   Final (median of 4 turned copies): "
                 f"diameter {final['width_mm']:.2f} mm, length {final['length_mm']:.2f} mm", fontsize=13, y=0.995)
    if path is not None:
        fig.savefig(path, dpi=95, bbox_inches="tight")
    return fig


def roughness_check(out_dir, cuts, key, px_to_mm, path=None):
    """Check 1 on a few cuts: mask, smoothed copy, changed pixels, roughness score."""
    fig, ax = plt.subplots(len(cuts), 4, figsize=(17, 4.4 * len(cuts)), squeeze=False)
    for i, cut in enumerate(cuts):
        p = Piece(out_dir, cut, key, px_to_mm)
        pad = int(1.5 * K) + 3
        mp_ = np.pad(p.m, pad)
        sm = cv2.GaussianBlur(binary_fill_holes(mp_).astype(np.float32), (0, 0), 0.5 * K) > 0.5
        diff = np.logical_xor(mp_, sm)
        score = measure.outline_roughness(p.m, K)
        _frame_panel(ax[i, 0], p, f"cut {cut}: the {key.upper()} mask on the video")
        ext = (0, mp_.shape[1] / K, mp_.shape[0] / K, 0)
        ax[i, 1].imshow(mp_, cmap="gray", extent=ext); ax[i, 1].set_title("1. The mask (in mm)", fontsize=10)
        ax[i, 2].imshow(sm, cmap="gray", extent=ext)
        ax[i, 2].set_title("2. A smoothed copy: outline blurred over\n~0.5 mm, holes filled", fontsize=10)
        show = np.zeros(mp_.shape + (3,), np.float32)
        show[sm] = 0.35
        show[diff] = (1.0, 0.15, 0.15)
        ax[i, 3].imshow(show, extent=ext)
        bad = score > measure.MAX_ROUGHNESS_PCT
        ax[i, 3].set_title(f"3. Red = pixels that changed: {diff.sum() / K**2:.1f} mm²\nROUGHNESS = {score:.1f} % "
                           f"of the area -> {'DROP (ragged)' if bad else 'keep'}", fontsize=10,
                           color="red" if bad else "green")
    fig.suptitle(f"Check 1: ragged outline. A clean mask hardly changes when smoothed; a speckled or torn one does. "
                 f"Limit: {measure.MAX_ROUGHNESS_PCT} %", fontsize=13, y=1.0)
    plt.tight_layout()
    if path is not None:
        fig.savefig(path, dpi=85, bbox_inches="tight")
    return fig


def tube_check(out_dir, cuts, key, px_to_mm, path=None):
    """Check 2 on a few cuts: the ideal tube vs the mask, and the tube fill number."""
    fig, ax = plt.subplots(len(cuts), 3, figsize=(14, 4.6 * len(cuts)), squeeze=False)
    lo_, hi_ = measure.TUBE_FILL_RANGE
    for i, cut in enumerate(cuts):
        p = Piece(out_dir, cut, key, px_to_mm)
        res = measure.measure_mask(p.mask, p.origin, px_to_mm)
        c = measure.centerline(p.m, K)
        tube = np.zeros(p.m.shape, np.uint8)
        cv2.polylines(tube, [np.round(c["path"][:, ::-1]).astype(np.int32)], False, 1,
                      thickness=2 * int(round(c["r_med"])) + 1)
        tube = tube > 0
        _frame_panel(ax[i, 0], p, f"cut {cut}: the {key.upper()} mask on the video")
        show = np.zeros(p.m.shape + (3,), np.float32)
        show[p.m & tube] = (0.75, 0.75, 0.75)
        show[p.m & ~tube] = (1.0, 0.15, 0.15)
        show[~p.m & tube] = (0.15, 0.35, 1.0)
        ax[i, 1].imshow(show, extent=(0, p.m.shape[1] / K, p.m.shape[0] / K, 0))
        ax[i, 1].plot(c["path"][:, 1] / K, c["path"][:, 0] / K, "y-", lw=1.5)
        ax[i, 1].set_title("The ideal tube: the path (yellow), as thick as\nthe measured diameter. grey = mask inside "
                           "the tube,\nred = extra mask outside it, blue = tube not filled", fontsize=10)
        fill = res.get("tube_fill", np.nan)
        verdict = ("DROP (extra area: merged piece)" if fill > hi_ else
                   "DROP (missing area: cut-off piece)" if fill < lo_ else "keep")
        ax[i, 2].axis("off")
        ax[i, 2].text(0.02, 0.55,
                      f"area            {res['area_mm2']:.0f} mm²\nlength          {res['length_mm']:.1f} mm\n"
                      f"diameter      {res['width_mm']:.2f} mm\n"
                      f"length x diameter = {res['length_mm'] * res['width_mm']:.0f} mm²\n\n"
                      f"TUBE FILL = area / (length x diameter)\n          = {fill:.2f}\n\n"
                      f"single tubes: 0.88 - 1.03\nallowed: {lo_:.2f} - {hi_:.2f}\n\n-> {verdict}",
                      fontsize=12, family="monospace", va="center",
                      color="red" if verdict.startswith("DROP") else "green")
    fig.suptitle("Check 2: does the mask look like ONE tube? Compare its area with length x diameter", fontsize=13, y=1.0)
    plt.tight_layout()
    if path is not None:
        fig.savefig(path, dpi=85, bbox_inches="tight")
    return fig


def checks_histograms(pieces, key, title="", path=None):
    """The three drop scores for many pieces, with their limits."""
    x = pieces[pieces["key"] == key]
    fig, ax = plt.subplots(1, 3, figsize=(19, 4.4))
    ax[0].hist(x["roughness_pct"].clip(upper=12), bins=np.arange(0, 12.25, 0.25), color="0.4")
    ax[0].axvline(measure.MAX_ROUGHNESS_PCT, color="r", ls="--", lw=2, label=f"limit {measure.MAX_ROUGHNESS_PCT} %")
    ax[0].set(xlabel="roughness (% of area)", ylabel=f"{key.upper()} masks", title="Check 1: roughness")
    ax[0].legend()
    ax[1].hist(x["tube_fill"].clip(0.6, 1.5), bins=np.arange(0.6, 1.52, 0.02), color="0.4")
    for v in measure.TUBE_FILL_RANGE:
        ax[1].axvline(v, color="r", ls="--", lw=2)
    ax[1].set(xlabel="tube fill = area / (length x diameter)", ylabel=f"{key.upper()} masks",
              title="Check 2: tube fill (allowed between the red lines)")
    ax[2].hist(x["shape_stability_pct"].clip(upper=12), bins=np.arange(0, 12.25, 0.25), color="0.4")
    ax[2].axvline(measure.MAX_STABILITY_PCT, color="r", ls="--", lw=2, label=f"limit {measure.MAX_STABILITY_PCT} %")
    ax[2].set(xlabel="change between 4 turned copies (%)", ylabel=f"{key.upper()} masks",
              title="Check 3: does the measurement depend on direction?")
    ax[2].legend()
    fig.suptitle(title, fontsize=12)
    plt.tight_layout()
    if path is not None:
        fig.savefig(path, dpi=85, bbox_inches="tight")
    return fig


def dropped_gallery(pieces, results_dir, video, key, max_n=28, path=None):
    """Every mask of one video that a shape check dropped, with the reason."""
    x = pieces[(pieces["video"] == video) & (pieces["key"] == key) & (pieces["mask_problem"].fillna("") != "")]
    x = x.sort_values(["mask_problem", "roughness_pct"], ascending=[True, False]).head(max_n)
    if not len(x):
        print(f"No {key.upper()} masks dropped in {video}.")
        return None
    ncol = 7
    nrow = int(np.ceil(len(x) / ncol))
    fig, ax = plt.subplots(nrow, ncol, figsize=(2.6 * ncol, 2.9 * nrow), squeeze=False)
    for a in ax.ravel():
        a.axis("off")
    short = {"ragged_outline": "ragged {r:.1f}%", "extra_area_merged_piece": "merged, fill {f:.2f}",
             "missing_area_cut_off_piece": "cut off, fill {f:.2f}", "unstable_measurement": "unstable {s:.1f}%"}
    for a, (_, r) in zip(ax.ravel(), x.iterrows()):
        a.imshow(Image.open(Path(results_dir) / video / "overlays" / f"cut{int(r.cut):04d}_{key}_f{int(r.frame):05d}.jpg"))
        a.set_title(f"cut {int(r.cut)}: " + short[r.mask_problem].format(r=r.roughness_pct, f=r.tube_fill,
                                                                         s=r.shape_stability_pct), fontsize=9)
    fig.suptitle(f"{video}: {key.upper()} masks dropped by the shape checks", fontsize=12)
    plt.tight_layout()
    if path is not None:
        fig.savefig(path, dpi=80, bbox_inches="tight")
    return fig


def reference_grid(videos, results_dir, mappings, ncol=5, half=170, path=None):
    """One small panel per video: clean die, the refined points (+) and where the mm map puts them (o)."""
    from . import calib
    from .viz import POINT_COLOURS, brighten
    ref = calib.die_map_mm()
    n = len(videos)
    nrow = int(np.ceil(n / ncol))
    fig, ax = plt.subplots(nrow, ncol, figsize=(3.6 * ncol, 3.4 * nrow), squeeze=False)
    for a in ax.ravel():
        a.axis("off")
    for a, v in zip(ax.ravel(), videos):
        v = Path(v)
        d = Path(results_dir) / v.stem
        clicks = json.loads((d / "clicks.json").read_text())
        pts = {**clicks.get("clicked", {}), **clicks.get("refined", {})}
        img = brighten(np.asarray(Image.open(d / "clean_die.png").convert("RGB")), 1.7)
        cx, cy = pts["hole_A"]
        a.imshow(img); a.set_xlim(cx - half * 0.6, cx + half * 1.4); a.set_ylim(cy + half * 0.6, cy - half * 1.4)
        mp_, name = mappings.get(v.stem, (None, "none"))
        for k, p in pts.items():
            a.plot(*p, "+", color=POINT_COLOURS.get(k, "m"), ms=11, mew=2)
        title = f"{v.stem}: {name}"
        if mp_ is not None:
            for k in ("hole_A", "hole_B", "bolt_near", "bolt_far"):
                q = calib.apply(mp_["mm_to_px"], ref[k])
                a.plot(*q, "o", mfc="none", mec=POINT_COLOURS.get(k, "m"), ms=13, mew=1.5)
            checks = [f"{k.replace('bolt_', 'b.')} {mp_['offset_mm'][k]:.1f}" for k in mp_["offset_mm"]
                      if k not in mp_["points_used"]]
            title += f"\n{mp_['px_per_mm']:.3f} px/mm" + (f" | check: {', '.join(checks)} mm" if checks else "")
        a.set_title(title, fontsize=8)
    fig.suptitle("Reference points of every video: + = refined points, o = where the mm map puts them "
                 "(check = distance between a point not used for the scale and the map)", fontsize=11)
    plt.tight_layout()
    if path is not None:
        fig.savefig(path, dpi=80, bbox_inches="tight")
    return fig

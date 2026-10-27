"""All pictures shown in the notebook (and saved to the results folder)."""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Circle
from PIL import Image, ImageDraw

from .calib import apply, die_map_mm, R_HOLE_MM, R_BOLT_MM, POINT_NAMES
from .track import touches_hole

GREEN = (80, 255, 80)
RED = (255, 60, 60)
BLUE = (60, 200, 255)
POINT_COLOURS = {"hole_A": "#ff4040", "bolt_near": "#ffd000", "bolt_far": "#ff9000",
                 "hole_B": "#40c0ff", "plate_centre": "#ffffff"}


def overlay(img, mask, colour, alpha=0.55):
    """Colour a mask on top of an image (both crop-sized). Returns uint8 RGB."""
    out = np.asarray(img).astype(np.float32).copy()
    if mask is not None:
        out[mask] = (1 - alpha) * out[mask] + alpha * np.array(colour, np.float32)
    return out.astype(np.uint8)


def brighten(img, gain=1.8):
    a = np.asarray(img).astype(np.float32)
    return np.clip((a - a.min()) * gain, 0, 255).astype(np.uint8)


def _save(fig, path):
    if path is not None:
        fig.savefig(path, dpi=110, bbox_inches="tight")


# ---------------------------------------------------------------- frames and crops
def crop_explainer(frame, hole_xy, crop_origin, crop_size, path=None):
    """Full frame -> analysis crop -> close-up of the hole, with every window drawn."""
    ox, oy = crop_origin
    hx, hy = hole_xy[0] - ox, hole_xy[1] - oy
    crop = frame[oy:oy + crop_size, ox:ox + crop_size]
    fig, ax = plt.subplots(1, 3, figsize=(17, 5.6), gridspec_kw=dict(width_ratios=[1.75, 1, 1]))
    ax[0].imshow(frame)
    ax[0].add_patch(Rectangle((ox, oy), crop_size, crop_size, fill=False, ec="cyan", lw=2))
    ax[0].plot(*hole_xy, "r+", ms=14, mew=2)
    ax[0].set_title(f"1. Full camera frame ({frame.shape[1]} x {frame.shape[0]} px)\n"
                    f"cyan = the {crop_size} x {crop_size} px window we keep (digital zoom)")
    ax[1].imshow(crop)
    ax[1].add_patch(Rectangle((hx - 40, hy - 40), 80, 80, fill=False, ec="yellow", lw=1.5, ls="--"))
    ax[1].add_patch(Rectangle((hx - 35, hy - 35), 70, 70, fill=False, ec="orange", lw=1))
    ax[1].add_patch(Rectangle((hx - 8, hy - 8), 16, 16, fill=False, ec="white", lw=1.5))
    ax[1].set_title("2. The crop given to SAM 2 (hole A in the centre)\n"
                    "yellow dashed = 80 px colour window, orange = box limit,\nwhite = 16 px 'is it attached?' square",
                    fontsize=9)
    z = 50
    zc = crop[int(hy) - z:int(hy) + z, int(hx) - z:int(hx) + z]
    ax[2].imshow(zc, extent=(-z, z, z, -z))
    ax[2].add_patch(Rectangle((-8, -8), 16, 16, fill=False, ec="white", lw=1.5))
    ax[2].add_patch(Circle((0, 0), 10, fill=False, ec="red", lw=1.2))
    ax[2].set_title("3. Close-up of hole A (100 x 100 px)\nred circle = where the 'blink' is measured\n"
                    "white = attachment square", fontsize=9)
    ax[2].set_xlabel("px from hole centre")
    for a in ax[:2]:
        a.set_xticks([]); a.set_yticks([])
    plt.tight_layout()
    _save(fig, path)
    return fig


def cycle_strip(crops, event, hole_xy, n=12, half=90, path=None):
    """One knife cycle at hole A: frames event-1 ... event+n-2, close-up."""
    hx, hy = int(round(hole_xy[0])), int(round(hole_xy[1]))
    frames = list(range(event - 1, event - 1 + n))
    fig, ax = plt.subplots(2, n // 2, figsize=(2.6 * n // 2, 6.0), layout="constrained")
    for a, f in zip(ax.ravel(), frames):
        if 0 <= f < len(crops):
            a.imshow(np.asarray(Image.open(crops[f]))[hy - half:hy + half, hx - half:hx + half])
            a.add_patch(Circle((half, half), 6, fill=False, ec="white", lw=1))
        lab = "event (knife passes)" if f == event else ("head-on rope end" if f == event + 1 else "")
        a.set_title(f"frame {f}  (e{f - event:+d})\n{lab}", fontsize=8)
        a.axis("off")
    fig.suptitle("One knife cycle at hole A (e = the detected cut event)")
    _save(fig, path)
    return fig


def signal_figure(s, fps, cuts, path=None):
    ev = cuts["events"]
    fig, ax = plt.subplots(1, 3, figsize=(17, 3.8))
    n1 = int(round(fps))
    t = np.arange(len(s)) / fps
    ax[0].plot(t[:n1], s[:n1], lw=1, color="#b8860b")
    e1 = ev[ev < n1]
    ax[0].plot(t[e1], s[e1], "r.", ms=9, label="detected cut events")
    ax[0].set(xlabel="time (s), first second only", ylabel="yellowness at hole A",
              title="The 'blink': yellowness at the hole, frame by frame")
    ax[0].legend(loc="lower right", fontsize=8)
    band = (cuts["freqs"] > 1) & (cuts["freqs"] < 100)
    ax[1].plot(cuts["freqs"][band], cuts["spectrum"][band], lw=1)
    peak = cuts["spectrum"][np.argmin(np.abs(cuts["freqs"] - cuts["cut_hz"]))]
    ax[1].plot(cuts["cut_hz"], peak, "o", mfc="none", mec="r", ms=12, mew=2,
               label=f"cut rhythm {cuts['cut_hz']:.2f}/s\n(peaks at 2x and 3x are its echoes)")
    ax[1].legend(fontsize=8)
    ax[1].set(xlabel="frequency (blinks per second)", ylabel="strength",
              title=f"Blink rhythm = {cuts['cut_hz']:.2f} cuts/s (no knife speed needed)")
    gaps = np.diff(ev)
    ax[2].hist(gaps, bins=np.arange(gaps.min() - 0.5, gaps.max() + 1.5) if len(gaps) else 10, color="#888")
    ax[2].set(xlabel="frames between consecutive cuts", ylabel="count",
              title=f"{len(ev)} cuts found in {len(s) / fps:.1f} s")
    plt.tight_layout()
    _save(fig, path)
    return fig


# ---------------------------------------------------------------- prompting and tracking
def prompt_steps_figure(img, steps, first_mask, hole_xy, view=(60, 300), path=None):
    """The 4 steps of the automatic prompt on the head-on frame."""
    a, b = view
    hx, hy = hole_xy
    box = steps["box"]
    fig, ax = plt.subplots(1, 4, figsize=(17, 4.8))
    ax[0].imshow(img); ax[0].add_patch(Circle((hx, hy), 4, fill=False, ec="white"))
    ax[0].set_title("1. Head-on frame: a new rope end\nappears on hole A (white circle)")
    ax[1].imshow(steps["yellow_map"], cmap="inferno")
    ax[1].set_title("2. 'Yellowness' map\n(red minus blue colour)")
    bl = np.asarray(img).astype(np.float32) * 0.45
    if steps["blob"] is not None:
        bl[steps["blob"]] = 255
    ax[2].imshow(bl.astype(np.uint8))
    ax[3].imshow(overlay(img, first_mask, GREEN))
    for k in (2, 3):
        if box is not None:
            ax[k].add_patch(Rectangle((box[0], box[1]), box[2] - box[0], box[3] - box[1], fill=False, ec="cyan", lw=2))
    ax[2].set_title(f"3. Yellow blob on the hole (threshold {steps['threshold']:.0f})\n"
                    "-> the computer draws the cyan box")
    ax[3].set_title("4. Box given to SAM 2\n-> SAM 2 returns the mask (green)")
    for x in ax:
        x.set_xlim(a, b); x.set_ylim(b, a); x.set_xticks([]); x.set_yticks([])
    plt.tight_layout()
    _save(fig, path)
    return fig


def frame_colour(f, result, hole_xy):
    """Colour and label for one tracked frame."""
    m = result["masks"].get(f)
    if f == result["red_frame"]:
        return RED, "RED: cut moment (last attached)"
    if f == result["blue_frame"]:
        return BLUE, "BLUE: first detached frame"
    if m is not None and touches_hole(m, hole_xy):
        return GREEN, "green: attached"
    return (200, 200, 200), "after detachment (ignored)" if result["blue_frame"] and f > result["blue_frame"] else "lost"


def track_strip(crops, result, f0, hole_xy, extra=2, view=(60, 300), path=None, title=None):
    """Every tracked frame from the head-on frame to 2 frames after detachment."""
    last = (result["blue_frame"] or max(result["masks"])) + extra
    frames = [f for f in range(f0, last + 1) if f in result["masks"]]
    ncol = min(7, len(frames))
    nrow = int(np.ceil(len(frames) / ncol))
    fig, ax = plt.subplots(nrow, ncol, figsize=(2.6 * ncol, 3.1 * nrow), squeeze=False, layout="constrained")
    a, b = view
    hx, hy = hole_xy
    for x in ax.ravel():
        x.axis("off")
    for x, f in zip(ax.ravel(), frames):
        col, lab = frame_colour(f, result, hole_xy)
        img = np.asarray(Image.open(crops[f]))
        show = overlay(img, result["masks"][f], col) if col != (200, 200, 200) else img
        x.imshow(show)
        x.add_patch(Rectangle((hx - 8, hy - 8), 16, 16, fill=False, ec="white", lw=1))
        x.set_xlim(a, b); x.set_ylim(b, a)
        x.set_title(f"+{f - f0} (frame {f})\n{lab}", fontsize=8,
                    color={RED: "red", BLUE: "tab:blue", GREEN: "green"}.get(col, "gray"))
    fig.suptitle(title or "SAM 2 follows the piece: green -> RED (cut) -> BLUE (detached). "
                 "White square = 16 px attachment test")
    _save(fig, path)
    return fig


def tracking_gif(crops, result, f0, hole_xy, path, extra=4, scale=2, ms=300, view=(60, 300)):
    """Slow animation of one tracked cut (saved as GIF)."""
    a, b = view
    last = min(max(result["masks"]), (result["blue_frame"] or max(result["masks"])) + extra)
    tiles = []
    for f in range(f0, last + 1):
        col, lab = frame_colour(f, result, hole_xy)
        img = np.asarray(Image.open(crops[f]))
        show = overlay(img, result["masks"].get(f), col) if col != (200, 200, 200) else img
        t = Image.fromarray(show[a:b, a:b]).resize(((b - a) * scale, (b - a) * scale), Image.LANCZOS)
        d = ImageDraw.Draw(t)
        hx, hy = (hole_xy[0] - a) * scale, (hole_xy[1] - a) * scale
        d.rectangle([hx - 8 * scale, hy - 8 * scale, hx + 8 * scale, hy + 8 * scale], outline=(255, 255, 255))
        d.rectangle([0, 0, t.width, 22], fill=(0, 0, 0))
        d.text((6, 5), f"frame {f} (+{f - f0})  {lab}", fill=(255, 255, 0))
        tiles.append(t)
    tiles[0].save(path, save_all=True, append_images=tiles[1:], duration=ms, loop=0)
    return path


def gallery(crops, rows, hole_xy, half=110, path=None):
    """Red and blue masks side by side for several cuts. rows: list of dicts from the cut table."""
    n = len(rows)
    if n == 0:
        return None
    fig, ax = plt.subplots(2, n, figsize=(2.3 * n, 5.4), squeeze=False, layout="constrained")
    hx, hy = int(round(hole_xy[0])), int(round(hole_xy[1]))
    for j, r in enumerate(rows):
        for i, (key, col) in enumerate((("red", RED), ("blue", BLUE))):
            f = int(r[f"{key}_frame"])
            m = np.asarray(Image.open(r[f"{key}_mask_file"])) > 0
            img = overlay(np.asarray(Image.open(crops[f])), m, col)
            ax[i, j].imshow(img[hy - half:hy + half, hx - half:hx + half])
            ax[i, j].set_title(f"cut {int(r['cut'])} {key}\nframe {f}", fontsize=8)
            ax[i, j].axis("off")
    fig.suptitle("Random kept cuts: top = RED (cut moment), bottom = BLUE (just detached)")
    _save(fig, path)
    return fig


def measurement_hists(table, cols, path=None):
    keep = table[table["keep"]]
    fig, ax = plt.subplots(1, len(cols), figsize=(4.2 * len(cols), 3.4), squeeze=False)
    for a, c in zip(ax[0], cols):
        for key, col in (("red", "tab:red"), ("blue", "tab:blue")):
            used = keep
            if f"{key}_mask_problem" in keep:   # leave out masks dropped by the shape checks
                used = keep[keep[f"{key}_mask_problem"].fillna("").eq("")]
            v = used[f"{key}_{c}"].dropna()
            if len(v):
                a.hist(v, bins=25, alpha=0.5, color=col, label=f"{key}: {v.mean():.2f} ± {v.std():.2f}")
        a.set(title=c, ylabel="pieces"); a.legend(fontsize=8)
    plt.tight_layout()
    _save(fig, path)
    return fig


# ---------------------------------------------------------------- calibration
def calibration_figure(image, clicked, refined, mapping, view_half=230, path=None):
    """The die-plate map drawn back onto the video, to check the mapping by eye."""
    ref = die_map_mm()
    mm_to_px = mapping["mm_to_px"]
    a = refined["hole_A"]
    fig, ax = plt.subplots(1, 2, figsize=(16, 7.5))
    for k, x in enumerate(ax):
        x.imshow(brighten(image) if k == 1 else image)
        th = np.linspace(0, 2 * np.pi, 200)
        for r, ls in ((R_HOLE_MM, "--"), (R_BOLT_MM, ":")):
            ring = apply(mm_to_px, np.stack([r * np.cos(th), r * np.sin(th)], 1))
            x.plot(ring[:, 0], ring[:, 1], ls, color="w", lw=0.8, alpha=0.7)
        for name, p_mm in ref.items():
            p = apply(mm_to_px, p_mm)
            used = name in mapping["points_used"]
            x.plot(*p, "o", mfc="none", mec=POINT_COLOURS[name], ms=12 if used else 16, mew=2,
                   ls="none")
            txt = name + (" (sets the scale)" if used else " (map position)")
            if name in mapping["offset_mm"] and (not used or len(mapping["points_used"]) > 2):
                txt += f"\nclick is {mapping['offset_mm'][name]:.1f} mm away"
            x.annotate(txt, p, xytext=(10, -10), textcoords="offset points", color=POINT_COLOURS[name], fontsize=8)
        for name, p in clicked.items():
            if p is not None:
                x.plot(*p, "x", color="magenta", ms=7)
        # 5 mm ruler next to hole A, along the mm x axis
        ruler = apply(mm_to_px, np.array([ref["hole_A"] + [3, 6], ref["hole_A"] + [8, 6]]))
        x.plot(ruler[:, 0], ruler[:, 1], "-", color="lime", lw=3)
        x.annotate("5 mm", ruler[1], xytext=(4, 4), textcoords="offset points", color="lime", fontsize=9)
        x.set_xlim(a[0] - view_half, a[0] + view_half * 1.3)
        x.set_ylim(a[1] + view_half, a[1] - view_half * 1.2)
        x.set_xticks([]); x.set_yticks([])
    lo, hi = mapping["px_per_mm_range"]
    spread = "same in every direction" if hi - lo < 0.01 else f"from {lo:.2f} to {hi:.2f} depending on direction"
    ax[0].set_title(f"Die-plate map drawn on the video (x = clicks / refined points, circles = map)\n"
                    f"scale {mapping['px_per_mm']:.2f} px/mm ({spread})")
    ax[1].set_title("Same, brightened. Check: do the circles sit on the real holes and bolts?\n"
                    "Dashed ring = 31 mm (die holes), dotted ring = 45 mm (bolts)")
    plt.tight_layout()
    _save(fig, path)
    return fig

"""Finding the cuts, and finding the exact centre of die hole A.

Idea in plain words
-------------------
Watch a tiny circle (radius 10 px) right on die hole A and measure how *yellow* it is
in every frame (red minus blue colour). Each time the knife passes, the hole is
covered for a moment and then a fresh yellow rope end pops out, so this "yellowness"
blinks once per cut. We find:

* the blink rhythm (cuts per second) from the signal itself, so we do not need to
  know the knife speed;
* one *event* per cut: the frame where the yellowness rises fastest. The frame right
  after the event (event + 1) shows the new rope end head-on, as a small round disc.
"""
import numpy as np
from scipy import ndimage as ndi
from scipy import signal as sps

from .video import read_crops


def yellowness(img):
    """Red minus blue, as float. Yellow product is high, the dark steel die is low."""
    a = np.asarray(img).astype(np.float32)
    return a[..., 0] - a[..., 2]


def hole_signal(frames, cx, cy, radius=10):
    """Mean yellowness inside a small disc at (cx, cy), for every frame. frames: (n, H, W, 3)."""
    h, w = frames.shape[1:3]
    yy, xx = np.mgrid[0:h, 0:w]
    disc = (xx - round(cx)) ** 2 + (yy - round(cy)) ** 2 <= radius ** 2
    sub = frames[:, disc].astype(np.float32)
    return (sub[..., 0] - sub[..., 2]).mean(1)


def cut_rhythm(s, fps, fmin=3.0, fmax=80.0):
    """Cuts per second: the strongest repeating frequency in the (de-trended) signal."""
    x = s - ndi.uniform_filter1d(s, 61)
    x = x - x.mean()
    spec = np.abs(np.fft.rfft(x * np.hanning(len(x))))
    freqs = np.fft.rfftfreq(len(x), 1 / fps)
    band = (freqs > fmin) & (freqs < fmax)
    return float(freqs[band][spec[band].argmax()]), freqs, spec


def detect_events(s, period_frames):
    """One event per cut: the frame where yellowness at the hole rises fastest."""
    ds = np.gradient(ndi.gaussian_filter1d(s, 1.0))
    ev, _ = sps.find_peaks(ds, distance=max(2, int(0.6 * period_frames)),
                           prominence=np.std(ds) * 0.8)
    return ev


def find_cuts(s, fps):
    """Rhythm + events in one call. Returns dict(cut_hz, period_frames, events)."""
    hz, freqs, spec = cut_rhythm(s, fps)
    period = fps / hz
    return dict(cut_hz=hz, period_frames=period, events=detect_events(s, period), freqs=freqs, spectrum=spec)


def headon_blob(frame, cx, cy, win=35):
    """The yellow blob sitting on (cx, cy) in a head-on frame (a fresh round rope end).

    Returns its centroid (in frame coordinates), size and roundness, or None.
    """
    cx, cy = int(round(cx)), int(round(cy))
    yel = ndi.gaussian_filter(yellowness(frame), 1.2)
    c = yel[cy - win:cy + win, cx - win:cx + win]
    if c.shape != (2 * win, 2 * win):
        return None
    thr = 0.5 * (np.percentile(c, 5) + np.percentile(c, 99))
    lab, n = ndi.label(c > thr)
    k = lab[win, win]
    if k == 0:
        ys, xs = np.nonzero(lab)
        if len(xs) == 0:
            return None
        d = np.hypot(xs - win, ys - win)
        if d.min() > 6:
            return None
        k = lab[ys[d.argmin()], xs[d.argmin()]]
    m = ndi.binary_fill_holes(lab == k)
    area = int(m.sum())
    if area < 30:
        return None
    perim = int((m ^ ndi.binary_erosion(m)).sum())
    circ = float(4 * np.pi * area / (perim ** 2 + 1e-9) * 0.79)  # rough pixel-perimeter correction
    touches = bool(m[0].any() or m[-1].any() or m[:, 0].any() or m[:, -1].any())
    ys, xs = np.nonzero(m)
    return dict(x=cx - win + xs.mean(), y=cy - win + ys.mean(), area=area,
                diam_px=float(2 * np.sqrt(area / np.pi)), circularity=circ,
                touches_border=touches, other_blobs=int(n - 1))


def refine_hole(video, click_xy, fps, half=60, iterations=3):
    """Exact centre of hole A from a rough click (or a default position).

    Reads a small window (2*half px) around the click for the whole clip, finds the
    cuts, and takes the median centre of all clean head-on rope ends. Repeated three
    times, so a click that is off by up to ~25 px still converges to the same centre
    (tested: clicks 15 px apart all end within 0.05 px of each other).
    """
    x0, y0 = int(round(click_xy[0])) - half, int(round(click_xy[1])) - half
    R = read_crops(video, x0, y0, 2 * half)
    cx, cy = float(half), float(half)
    history = [(click_xy[0], click_xy[1])]
    used = []
    for _ in range(iterations):
        s = hole_signal(R, cx, cy)
        cuts = find_cuts(s, fps)
        used = []
        for e in cuts["events"]:
            if e + 1 >= len(R):
                continue
            b = headon_blob(R[e + 1], cx, cy, win=min(35, half - 2))
            if b and b["circularity"] > 0.8 and not b["touches_border"] and b["other_blobs"] <= 1:
                used.append((b["x"], b["y"]))
        if len(used) < 5:
            break
        cx, cy = np.median(np.array(used), axis=0)
        history.append((x0 + cx, y0 + cy))
    return dict(hole_xy=(float(x0 + cx), float(y0 + cy)), n_rope_ends_used=len(used),
                spread_px=float(np.median(np.abs(np.array(used) - [cx, cy]))) if used else None,
                history=history)

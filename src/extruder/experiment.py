"""Experiment 1 (geometric, zero training): measure every RED/BLUE mask in mm and compare with the lab.

Plain words
-----------
1. Every kept mask of every video is measured (length along the bend, median diameter;
   see measure.py) and checked for problems (ragged, merged, cut off, unstable).
2. Per clip and per mask type (RED / BLUE): mean, SD, median and number of pieces used.
3. These video clip means are compared directly with the lab's clip means, for every clip.
   Nothing is learned from the lab data: the millimetres come only from the die-plate
   ruler (holes A and B, or the bolts). So every clip is a test clip.
4. Two variants are compared: with the bad masks dropped (main) and with every mask kept,
   to see what the shape checks are worth.
5. Errors are reported next to the noise floor (how precisely the lab's 5 pieces know their
   own clip mean), with bootstrap confidence intervals.
"""
import json
import multiprocessing as mp
import os
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from . import calib, measure

MEASURE_VERSION = "measure-v5"     # change when measure.py changes, to re-measure every mask
NOMINAL_PX_PER_MM = 3.5            # used only to draw masks on a grid when a video has no reference points
TARGETS = {"diameter": "width_mm", "length": "length_mm"}
KEYS = ("red", "blue")


# ------------------------------------------------------------------ reference points -> mm
def video_mapping(clicks):
    """Pixel -> mm mapping of one video from its saved (refined) points. Returns (mapping or None, name)."""
    pts = {**{k: v for k, v in clicks.get("clicked", {}).items()}, **clicks.get("refined", {})}
    use, name = calib.choose_reference(pts)
    if use is None:
        return None, name
    return calib.fit_mapping(pts, use=use), name


def nominal_mapping():
    s = 1 / NOMINAL_PX_PER_MM
    return np.array([[s, 0, 0], [0, s, 0]], float)


# ------------------------------------------------------------------ measuring all masks
def _init_worker():
    import cv2
    cv2.setNumThreads(1)


def _measure_one(task):
    mask_file, origin, px_to_mm = task
    from PIL import Image
    try:
        m = np.asarray(Image.open(mask_file)) > 0
        return measure.measure_mask(m, tuple(origin), np.array(px_to_mm))
    except Exception as e:  # a single bad mask must not stop the whole run
        return {"error": str(e)}


def measure_video(out_dir, px_to_mm, px_per_mm, reference, n_jobs=8, force=False, log=print):
    """Measure every RED and BLUE mask of the cuts kept by tracking. Cached in pieces.csv."""
    out_dir = Path(out_dir)
    table = pd.read_csv(out_dir / "cuts.csv")
    origin = json.loads((out_dir / "crop_frames" / "meta.json").read_text())["origin"]
    cfg = dict(version=MEASURE_VERSION, px_to_mm=np.round(px_to_mm, 9).tolist(), reference=reference,
               cuts_mtime=os.path.getmtime(out_dir / "cuts.csv"))
    cfg_path, cache = out_dir / "pieces.json", out_dir / "pieces.csv"
    if not force and cache.exists() and cfg_path.exists() and json.loads(cfg_path.read_text()) == cfg:
        return pd.read_csv(cache)
    rows, tasks = [], []
    for _, r in table[table["keep"]].iterrows():
        for key in KEYS:
            f = r.get(f"{key}_mask_file")
            if isinstance(f, str) and Path(f).exists():
                rows.append(dict(cut=int(r["cut"]), key=key, frame=int(r[f"{key}_frame"]), mask_file=f,
                                 frames_to_cut=r.get("frames_to_cut")))
                tasks.append((f, origin, np.asarray(px_to_mm).tolist()))
    # "spawn" starts clean worker processes; copying ("fork") a running notebook kernel that already
    # uses many threads (PyTorch, OpenCV) can leave the workers waiting forever
    ctx = mp.get_context("spawn")
    with ProcessPoolExecutor(max_workers=max(1, min(n_jobs, len(tasks))), mp_context=ctx,
                             initializer=_init_worker) as ex:
        results = list(ex.map(_measure_one, tasks, chunksize=8))
    for row, res in zip(rows, results):
        row.update(res)
    pieces = pd.DataFrame(rows)
    pieces["px_per_mm"] = px_per_mm
    pieces["reference"] = reference
    pieces.to_csv(cache, index=False)
    cfg_path.write_text(json.dumps(cfg))
    return pieces


def collect_pieces(per_video, clip_of):
    """All videos in one table. The drop decision is recomputed here, so new limits need no re-measuring."""
    frames = []
    for video, df in per_video.items():
        df = df.copy()
        df.insert(0, "video", video)
        df.insert(1, "clip", clip_of(video))
        df.insert(2, "setting", str(clip_of(video)).split("_")[0])
        frames.append(df)
    p = pd.concat(frames, ignore_index=True)
    p["mask_problem"] = [measure.mask_problem(r) if not isinstance(r.get("error"), str) else "measure_error"
                         for r in p.to_dict("records")]
    p["used"] = p["mask_problem"].eq("")
    # pixel-equivalent values (for method R0): mm under a scale-only mapping times px per mm
    for t, col in TARGETS.items():
        p[col.replace("_mm", "_px")] = p[col] * p["px_per_mm"]
    p.loc[p["reference"].eq("none"), list(TARGETS.values())] = np.nan
    return p


# ------------------------------------------------------------------ clip averages
def clip_table(pieces, labels):
    """Per clip and mask type: pieces used, mean / SD / median of diameter and length (mm and px)."""
    rows = []
    for (clip, key), g in pieces.groupby(["clip", "key"]):
        u = g[g["used"]]
        r = dict(clip=clip, key=key, setting=g["setting"].iloc[0], video=g["video"].iloc[0],
                 n_measured=len(g), n_used=len(u), reference=g["reference"].iloc[0],
                 px_per_mm=float(g["px_per_mm"].iloc[0]))
        for t, col in TARGETS.items():
            for src, tag in ((u, ""), (g, "_all")):
                v = src[col].dropna()
                r[f"video_{t}_mean{tag}"] = float(v.mean()) if len(v) else np.nan
                r[f"video_{t}_sd{tag}"] = float(v.std()) if len(v) > 1 else np.nan
            r[f"video_{t}_median"] = float(u[col].median()) if len(u) else np.nan
            r[f"video_{t}_px_mean"] = float(u[col.replace("_mm", "_px")].mean()) if len(u) else np.nan
        rows.append(r)
    ct = pd.DataFrame(rows)
    return ct.merge(labels, on=["clip", "setting"], how="left")


# ------------------------------------------------------------------ zero-training comparison
VARIANTS = {"bad masks dropped": "video_{t}_mean", "every mask kept": "video_{t}_mean_all"}
MAIN_VARIANT = "bad masks dropped"


def compare(ct):
    """Video clip mean vs lab clip mean for every clip, mask type, target and variant (nothing fitted)."""
    rows = []
    d = ct[ct["lab_diameter_mean"].notna()]
    for _, r in d.iterrows():
        for t in TARGETS:
            for variant, col in VARIANTS.items():
                rows.append(dict(variant=variant, key=r["key"], target=t, clip=r["clip"], setting=r["setting"],
                                 video=r[col.format(t=t)], lab=r[f"lab_{t}_mean"], lab_sd=r[f"lab_{t}_sd"],
                                 lab_se=r[f"lab_{t}_se"], video_sd=r[f"video_{t}_sd"], n_used=r["n_used"]))
    out = pd.DataFrame(rows)
    out["error"] = out["video"] - out["lab"]
    return out


def _bootstrap_ci(err, n=2000, seed=0):
    rng = np.random.default_rng(seed)
    if len(err) < 2:
        return np.nan, np.nan
    b = rng.choice(err, size=(n, len(err)), replace=True).mean(1)
    return float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))


def metrics(comp):
    """Error measures per variant, mask type and target, over all clips."""
    rows = []
    for (variant, key, t), g in comp.groupby(["variant", "key", "target"], sort=False):
        g = g.dropna(subset=["video", "lab"])
        if not len(g):
            continue
        e = g["error"].to_numpy()
        ae = np.abs(e)
        lo, hi = _bootstrap_ci(ae)
        ss_res, ss_tot = (e ** 2).sum(), ((g["lab"] - g["lab"].mean()) ** 2).sum()
        r = float(np.corrcoef(g["video"], g["lab"])[0, 1]) if len(g) > 2 else np.nan
        rows.append(dict(variant=variant, key=key, target=t, clips=len(g), MAE=ae.mean(), MAE_CI_low=lo,
                         MAE_CI_high=hi, RMSE=np.sqrt((e ** 2).mean()), MAPE=float((ae / g["lab"].abs()).mean() * 100),
                         bias=e.mean(), limits_of_agreement=1.96 * e.std(ddof=1) if len(e) > 1 else np.nan,
                         R2=float(1 - ss_res / ss_tot) if ss_tot > 0 else np.nan, correlation=r,
                         noise_floor=float(g["lab_se"].median()),
                         SD_MAE=float(np.abs(g["video_sd"] - g["lab_sd"]).mean())))
    return pd.DataFrame(rows)


# ------------------------------------------------------------------ figures
SET_COLOURS = {"901": "tab:blue", "902": "tab:orange", "903": "tab:green", "904": "tab:red"}


def plot_clips(ct, key, path=None):
    """Lab mean ± SD next to the video mean ± SD (ruler only, zero training), clip by clip."""
    d = ct[(ct["key"] == key) & ct["lab_diameter_mean"].notna()].sort_values("clip")
    fig, ax = plt.subplots(2, 1, figsize=(16, 8), sharex=True)
    x = np.arange(len(d))
    for a, t in zip(ax, TARGETS):
        a.errorbar(x - 0.12, d[f"lab_{t}_mean"], d[f"lab_{t}_sd"], fmt="o", color="k", capsize=3, label="lab (mean ± SD)")
        a.errorbar(x + 0.12, d[f"video_{t}_mean"], d[f"video_{t}_sd"], fmt="s", color="tab:blue" if key == "blue"
                   else "tab:red", capsize=3, label=f"video {key.upper()} (mean ± SD, zero training)")
        a.set_ylabel(f"{t} (mm)"); a.legend(fontsize=9); a.grid(alpha=0.3)
    ax[1].set_xticks(x); ax[1].set_xticklabels(d["clip"], rotation=60)
    fig.suptitle(f"Every clip: lab vs video ({key.upper()} masks, ruler only)", fontsize=13)
    plt.tight_layout()
    if path is not None:
        fig.savefig(path, dpi=90, bbox_inches="tight")
    return fig


def plot_errors(met, path=None):
    """Average error per clip, RED and BLUE, with and without dropping, next to the lab noise floor."""
    fig, ax = plt.subplots(1, 2, figsize=(15, 4.8))
    for a, t in zip(ax, TARGETS):
        m = met[met["target"] == t]
        variants = list(VARIANTS)
        for j, (key, col) in enumerate((("red", "tab:red"), ("blue", "tab:blue"))):
            mk = m[m["key"] == key].set_index("variant").reindex(variants)
            xs = np.arange(len(variants)) + (j - 0.5) * 0.38
            a.bar(xs, mk["MAE"], 0.36, color=col, alpha=0.8, label=key.upper())
            a.errorbar(xs, mk["MAE"], [mk["MAE"] - mk["MAE_CI_low"], mk["MAE_CI_high"] - mk["MAE"]],
                       fmt="none", ecolor="k", capsize=3, lw=1)
            for x, v in zip(xs, mk["MAE"]):
                a.text(x, v, f"{v:.2f}", ha="center", va="bottom", fontsize=9)
        nf = m["noise_floor"].median()
        a.axhline(nf, color="green", ls="--", lw=2, label=f"lab noise floor {nf:.2f} mm")
        a.set_xticks(np.arange(len(variants))); a.set_xticklabels(variants)
        a.set(ylabel="mean absolute error per clip (mm)", title=f"{t.capitalize()}: video vs lab (zero training)")
        a.legend(fontsize=9); a.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    if path is not None:
        fig.savefig(path, dpi=90, bbox_inches="tight")
    return fig


def plot_agreement(comp, key, variant=MAIN_VARIANT, path=None):
    """Video vs lab (left) and Bland-Altman (right), one point per clip, coloured by setting."""
    fig, ax = plt.subplots(2, 2, figsize=(14, 11))
    for row, t in enumerate(TARGETS):
        g = comp[(comp["variant"] == variant) & (comp["key"] == key) & (comp["target"] == t)].dropna(subset=["video"])
        a = ax[row, 0]
        for s, gs in g.groupby("setting"):
            a.errorbar(gs["lab"], gs["video"], xerr=gs["lab_se"], fmt="o", color=SET_COLOURS.get(s, "k"),
                       label=f"setting {s}", capsize=2)
        lim = [min(g["lab"].min(), g["video"].min()) * 0.95, max(g["lab"].max(), g["video"].max()) * 1.05]
        a.plot(lim, lim, "k--", lw=1, label="video = lab"); a.set_xlim(lim); a.set_ylim(lim)
        a.set(xlabel=f"lab {t} (mm, ± standard error)", ylabel=f"video {t} (mm)", title=f"{t}: video vs lab")
        a.legend(fontsize=8); a.grid(alpha=0.3)
        a = ax[row, 1]
        mean, diff = (g["video"] + g["lab"]) / 2, g["video"] - g["lab"]
        for s, idx in g.groupby("setting").groups.items():
            a.scatter(mean[idx], diff[idx], color=SET_COLOURS.get(s, "k"), label=f"setting {s}")
        bias, sd = diff.mean(), diff.std()
        for v, ls in ((bias, "-"), (bias + 1.96 * sd, "--"), (bias - 1.96 * sd, "--")):
            a.axhline(v, color="k", ls=ls, lw=1)
        a.axhline(0, color="0.6", lw=0.8)
        a.set(xlabel=f"average of video and lab {t} (mm)", ylabel="video − lab (mm)",
              title=f"Bland–Altman: bias {bias:+.2f} mm, 95 % limits {bias - 1.96 * sd:+.2f} to {bias + 1.96 * sd:+.2f}")
        a.grid(alpha=0.3)
    fig.suptitle(f"Zero training — {key.upper()} masks — {variant}", fontsize=13)
    plt.tight_layout()
    if path is not None:
        fig.savefig(path, dpi=90, bbox_inches="tight")
    return fig

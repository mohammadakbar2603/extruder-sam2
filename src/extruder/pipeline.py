"""The whole pipeline for one video, as reusable steps (the notebook calls these one by one).

Results for a video named <name> go to results/<name>/:

    clicks.json          your clicks and the refined reference points
    clean_die.png        median "clean die" picture used for clicking
    crop_frames/         the 360 x 360 crop of every frame (JPEG; SAM 2 reads these)
    masks/               cut0007_red_f01762.png, cut0007_blue_f01763.png, ... (black/white masks, crop pixels)
    overlays/            the same masks drawn on the image, to check by eye
    figures/             all figures shown in the notebook
    cuts.csv             one row per cut: frames, keep / reject reason, measurements for red and blue
    summary.json/.csv    mean +/- SD per measurement over the kept cuts
"""
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

from . import calib, cuts as cutsmod
from .measure import measure_mask
from .prompt import disc_box
from .track import track_cut
from .video import iter_crops, probe

DEFAULT_HOLE_A = (722, 488)   # where hole A usually is in our camera view (clip 0069)


# -------------------------------------------------------------- step 1: reference points
def load_clicks(path):
    path = Path(path)
    return json.loads(path.read_text()) if path.exists() else {}


def refine_points(video, info, clicked, log=print):
    """Hole A is snapped to its exact centre (from the rope ends); bolts and hole B are used as clicked."""
    refined = {}
    xy = lambda p: f"({float(p[0]):.1f}, {float(p[1]):.1f})"
    a = clicked.get("hole_A") or DEFAULT_HOLE_A
    if not clicked.get("hole_A"):
        log(f"No click on hole A: starting from the default position {DEFAULT_HOLE_A}.")
    r = cutsmod.refine_hole(video, a, info["fps"])
    refined["hole_A"] = r["hole_xy"]
    log(f"hole A: click {xy(a)} -> centre {xy(r['hole_xy'])} "
        f"(median of {r['n_rope_ends_used']} rope ends, typical spread {r['spread_px']:.1f} px)"
        if r["spread_px"] is not None else "hole A: could not refine; using the click as is.")
    for name in ("bolt_near", "bolt_far"):
        if clicked.get(name) is not None:
            # bolts are used exactly where you clicked: zoom in and click the centre of the head
            refined[name] = tuple(clicked[name])
            log(f"{name}: using the click {xy(clicked[name])}")
    if clicked.get("hole_B") is not None:
        refined["hole_B"] = tuple(clicked["hole_B"])
        log(f"hole_B: using the click {xy(clicked['hole_B'])}")
    return refined, r


# -------------------------------------------------------------- step 2: crop + cuts
def prepare_crops(video, out_dir, hole_xy, crop_size=360, log=print):
    """Cut the crop around hole A out of every frame, save as JPEGs, and measure the blink signal."""
    info = probe(video)
    out_dir = Path(out_dir)
    fdir = out_dir / "crop_frames"
    half = crop_size // 2
    origin = (int(round(hole_xy[0])) - half, int(round(hole_xy[1])) - half)
    hole_in_crop = (hole_xy[0] - origin[0], hole_xy[1] - origin[1])
    meta_path = fdir / "meta.json"
    meta = dict(video=Path(video).name, origin=origin, crop_size=crop_size)
    files = sorted(fdir.glob("*.jpg"))
    if meta_path.exists() and json.loads(meta_path.read_text()) == meta and len(files) >= info["n_frames"] - 2:
        log(f"Re-using {len(files)} crop frames already in {fdir}")
        sig = np.load(fdir / "signal.npy")
    else:
        fdir.mkdir(parents=True, exist_ok=True)
        for f in fdir.glob("*.jpg"):
            f.unlink()
        t = time.time()
        yy, xx = np.mgrid[0:crop_size, 0:crop_size]
        disc = (xx - round(hole_in_crop[0])) ** 2 + (yy - round(hole_in_crop[1])) ** 2 <= 10 ** 2
        sig = []
        for i, fr in enumerate(iter_crops(video, origin[0], origin[1], crop_size, info["width"], info["height"])):
            Image.fromarray(fr).save(fdir / f"{i:05d}.jpg", quality=95)
            sub = fr[disc].astype(np.float32)
            sig.append(float((sub[:, 0] - sub[:, 2]).mean()))
        sig = np.array(sig)
        np.save(fdir / "signal.npy", sig)
        meta_path.write_text(json.dumps(meta))
        files = sorted(fdir.glob("*.jpg"))
        log(f"Wrote {len(files)} crop frames in {time.time() - t:.0f} s to {fdir}")
    found = cutsmod.find_cuts(sig, info["fps"])
    log(f"Blink rhythm {found['cut_hz']:.2f} cuts/s -> {len(found['events'])} cuts in {info['duration_s']:.1f} s")
    return dict(info=info, origin=origin, hole_in_crop=hole_in_crop, frame_files=[str(f) for f in files],
                signal=sig, cuts=found)


# -------------------------------------------------------------- step 3: one cut
def process_cut(predictor, prep, k, event, device="cuda", n_track=16):
    """Automatic box on the head-on frame -> SAM 2 tracking -> red / blue frames."""
    files, hole = prep["frame_files"], prep["hole_in_crop"]
    f0 = int(event) + 1
    if f0 >= len(files):
        return dict(cut=k, event_frame=int(event), headon_frame=f0, keep=False, reason="end_of_video"), None
    img = np.asarray(Image.open(files[f0]).convert("RGB"))
    box = disc_box(img, hole)
    if box is None:
        return dict(cut=k, event_frame=int(event), headon_frame=f0, keep=False, reason="no_rope_end_at_hole"), None
    res = track_cut(predictor, files, f0, box, hole, device=device, n_track=n_track)
    row = dict(cut=k, event_frame=int(event), headon_frame=f0, red_frame=res["red_frame"],
               blue_frame=res["blue_frame"], timeline=res["timeline"],
               frames_to_cut=(res["red_frame"] - f0) if res["red_frame"] is not None else None,
               box=[float(v) for v in box])
    row.update(check_cut(res, prep))
    return row, res


def check_cut(res, prep, min_area=200, ratio=(0.5, 2.0), edge=2):
    """Automatic quality checks. Returns keep (True/False) and the reason if rejected."""
    red, blue = res["red_frame"], res["blue_frame"]
    if blue is None:
        return dict(keep=False, reason="no_detachment_within_window")
    if red is None:
        return dict(keep=False, reason="never_attached")
    if blue != red + 1:
        return dict(keep=False, reason="mask_lost_before_cut")
    mr, mb = res["masks"][red], res["masks"][blue]
    for m in (mr, mb):
        if m[:edge].any() or m[-edge:].any() or m[:, :edge].any() or m[:, -edge:].any():
            return dict(keep=False, reason="touches_crop_edge")
    if mr.sum() < min_area:
        return dict(keep=False, reason="too_small")
    q = mb.sum() / max(1, mr.sum())
    if not ratio[0] <= q <= ratio[1]:
        return dict(keep=False, reason="size_jump_red_to_blue")
    return dict(keep=True, reason="")


# -------------------------------------------------------------- step 4: save + measure
def save_and_measure(row, res, prep, out_dir, px_to_mm=None):
    """Save the red and blue masks (with frame numbers in the names) and measure both."""
    from .viz import overlay, RED, BLUE
    out_dir = Path(out_dir)
    (out_dir / "masks").mkdir(parents=True, exist_ok=True)
    (out_dir / "overlays").mkdir(parents=True, exist_ok=True)
    hx, hy = (int(round(v)) for v in prep["hole_in_crop"])
    for key, col in (("red", RED), ("blue", BLUE)):
        f = row.get(f"{key}_frame")
        if f is None or res is None:
            continue
        m = res["masks"][f]
        name = f"cut{row['cut']:04d}_{key}_f{f:05d}"
        Image.fromarray((m * 255).astype(np.uint8)).save(out_dir / "masks" / f"{name}.png")
        img = np.asarray(Image.open(prep["frame_files"][f]).convert("RGB"))
        Image.fromarray(overlay(img, m, col)[max(0, hy - 120):hy + 120, max(0, hx - 120):hx + 120]).save(
            out_dir / "overlays" / f"{name}.jpg", quality=90)
        row[f"{key}_mask_file"] = str(out_dir / "masks" / f"{name}.png")
        for k, v in measure_mask(m, prep["origin"], px_to_mm).items():
            row[f"{key}_{k}"] = v
    return row


def flag_outliers(table, col="red_area_px", k=4.0):
    """Reject kept cuts whose red area is far from the clip's typical value (median +/- 4 MAD)."""
    keep = table["keep"] & table[col].notna()
    v = table.loc[keep, col]
    if len(v) < 10:
        return table
    med = v.median()
    mad = (v - med).abs().median() * 1.4826 + 1e-9
    bad = keep & ((table[col] - med).abs() > k * mad)
    table.loc[bad, "keep"] = False
    table.loc[bad, "reason"] = "unusual_size_for_this_clip"
    return table


MEASURES = ["area_mm2", "length_mm", "width_mm", "width_max_mm", "width_mean_mm", "feret_mm", "straightness",
            "expansion", "area_px", "length_px", "width_px"]


def summarise(table, video_name, extra=None):
    """Mean +/- SD of every measurement over the kept cuts, for red and blue separately."""
    keep = table[table["keep"]]
    s = dict(video=video_name, cuts_detected=int(len(table)), cuts_kept=int(len(keep)))
    s.update({f"rejected_{r}": int(n) for r, n in table.loc[~table["keep"], "reason"].value_counts().items()})
    for key in ("red", "blue"):
        for m in MEASURES:
            c = f"{key}_{m}"
            if c in keep and keep[c].notna().any():
                s[f"{c}_mean"] = float(keep[c].mean())
                s[f"{c}_sd"] = float(keep[c].std())
    if extra:
        s.update(extra)
    return s


# -------------------------------------------------------------- everything for all cuts of one video
def run_cuts(predictor, prep, out_dir, px_to_mm=None, device="cuda", n_track=16, max_cuts=None, progress=True):
    """Track every cut of the video, save both masks per cut, measure them. Returns the cut table."""
    from tqdm.auto import tqdm
    events = prep["cuts"]["events"]
    idx = range(len(events)) if max_cuts is None else range(min(max_cuts, len(events)))
    rows = []
    for k in tqdm(idx, desc="cuts", disable=not progress):
        row, res = process_cut(predictor, prep, k, events[k], device=device, n_track=n_track)
        if res is not None:
            row = save_and_measure(row, res, prep, out_dir, px_to_mm)
        rows.append(row)
    table = flag_outliers(pd.DataFrame(rows))
    table.to_csv(Path(out_dir) / "cuts.csv", index=False)
    return table


def save_summary(table, out_dir, video_name, extra=None):
    s = summarise(table, video_name, extra)
    Path(out_dir, "summary.json").write_text(json.dumps(s, indent=1))
    pd.DataFrame([s]).to_csv(Path(out_dir, "summary.csv"), index=False)
    return s


def run_video(video, results_dir, predictor, crop_size=360, calibration_points=("bolt_near", "bolt_far"),
              device="cuda", n_track=16, max_cuts=None, log=print):
    """The whole pipeline for one video, using its saved clicks (results/<name>/clicks.json)."""
    video = Path(video)
    out_dir = Path(results_dir) / video.stem
    clicks = load_clicks(out_dir / "clicks.json")
    if not clicks.get("clicked"):
        log(f"[skip] {video.name}: no clicks saved yet. Open it in the notebook and click the points first.")
        return None
    info = probe(video)
    refined, _ = refine_points(video, info, clicks["clicked"], log=log)
    mapping = None
    try:
        mapping = calib.fit_mapping({**clicks["clicked"], **refined}, use=calibration_points)
    except ValueError as e:
        log(f"No mm conversion for {video.name}: {e}")
    prep = prepare_crops(video, out_dir, refined["hole_A"], crop_size, log=log)
    table = run_cuts(predictor, prep, out_dir, mapping["px_to_mm"] if mapping else None, device, n_track, max_cuts)
    extra = dict(cut_hz=prep["cuts"]["cut_hz"], px_per_mm=mapping["px_per_mm"] if mapping else None,
                 hole_A_px=list(refined["hole_A"]), crop_origin=list(prep["origin"]))
    return save_summary(table, out_dir, video.name, extra)

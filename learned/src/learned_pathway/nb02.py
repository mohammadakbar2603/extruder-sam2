"""Read this repository's notebook 02 results for notebook 03 (the learned pathway).

Notebook 02 (`external/extruder-sam2/notebooks/02_experiment1_geometric.ipynb`) leaves, per video,
`cuts.csv` (every knife cut: RED and BLUE frames, kept or not), `clicks.json` (the clicked and
refined reference points), `crop_frames/meta.json` (where the 360 x 360 crop sits in the frame) and
`masks/`; and `experiment1/pieces.csv`, one row per measured RED or BLUE mask with `used` (passed
the shape checks), `width_mm`, `length_mm` and `mask_file`.

This module adds what notebook 03 needs on top:
- mask paths that work on this computer (notebook 02 may have run on another one);
- **repeats**: a cut whose RED frame equals the RED frame of the cut before it is the same piece
  tracked twice, so it is dropped for RED and BLUE alike;
- each clip's pixel -> mm ruler, exactly as notebook 02 computes it;
- the lab's clip means, and the clips that have both lab values and pieces.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]  # the repository: learned/src/learned_pathway/nb02.py
if str(ROOT / "src") not in sys.path:  # the extruder package, also when it is not pip-installed
    sys.path.insert(0, str(ROOT / "src"))

from extruder import experiment, labels  # noqa: E402

RESULTS_DIR = ROOT / "results"                               # as in notebook 02
VIDEOS_DIR = ROOT / "data" / "Video from Trial Run 10.07.26"   # as in notebook 02
LAB_SHEET = ROOT / "data" / "Trial Run Data.xlsx"              # as in notebook 02
LAB_SHEET_NAME = "Oct 7 Final"
LAB_COLUMNS = dict(clip="Sl. No.", piece="Piece no.", length="Length of the Product (mm)",
                   diameter="Diameter of the Product (mm)")
KNIFE_RPM = {"901": 450, "902": 450, "903": 1005, "904": 750}  # knife speed per setting, from the run sheet


def local_mask_path(results_dir: Path, written_path: str) -> Path:
    """`results_dir/<clip>/masks/<file name>` for a mask path written on another computer."""
    p = Path(str(written_path))
    return Path(results_dir) / p.parent.parent.name / "masks" / p.name


def repeat_cuts(cuts: pd.DataFrame) -> set[int]:
    """Cuts whose RED frame equals the RED frame of the cut before them (in cut order, kept or not)."""
    c = cuts.sort_values("cut")
    red = c["red_frame"].to_numpy(float)
    same = np.r_[False, red[1:] == red[:-1]]  # NaN never equals NaN
    return {int(x) for x in c["cut"].to_numpy()[same]}


def load_pieces(results_dir: Path) -> pd.DataFrame:
    """Notebook 02's pieces table with `mask_path` (local), `repeat`, `origin_x` and `origin_y` added."""
    results_dir = Path(results_dir)
    p = pd.read_csv(results_dir / "experiment1" / "pieces.csv", dtype={"clip": str, "setting": str})
    p["mask_problem"] = p["mask_problem"].fillna("")
    p["mask_path"] = [str(local_mask_path(results_dir, f)) for f in p["mask_file"]]
    repeats, origin = {}, {}
    for clip in p["clip"].unique():
        repeats[clip] = repeat_cuts(pd.read_csv(results_dir / clip / "cuts.csv"))
        origin[clip] = json.loads((results_dir / clip / "crop_frames" / "meta.json").read_text())["origin"]
    p["repeat"] = [int(cut) in repeats[clip] for clip, cut in zip(p["clip"], p["cut"])]
    p["origin_x"] = [int(origin[c][0]) for c in p["clip"]]
    p["origin_y"] = [int(origin[c][1]) for c in p["clip"]]
    return p


def piece_set(pieces: pd.DataFrame, key: str, clips: Sequence[str]) -> pd.DataFrame:
    """The pieces every method uses: `key` masks of `clips` that passed the checks and are no repeat.

    A clip with fewer than two such pieces has no spread and no stable mean, so it is refused.
    Column `row` is the piece's position in `pieces`, which is also its row in the feature array.
    """
    pieces = pieces.reset_index(drop=True)  # rows by position, as `piece_features.build_features` stores them
    s = pieces[pieces["key"].eq(key) & pieces["used"].astype(bool) & ~pieces["repeat"].astype(bool)
               & pieces["clip"].isin(list(clips))]
    counts = s["clip"].value_counts()
    thin = sorted(c for c in clips if counts.get(c, 0) < 2)
    if thin:
        raise ValueError(f"{key} pieces: fewer than 2 used, non-repeated masks in clip(s) {thin}")
    return s.rename_axis("row").reset_index()  # `row`: the piece's position in `pieces` (and in its features)


def clip_ruler(results_dir: Path, clip: str) -> np.ndarray:
    """The clip's full-frame px -> mm mapping (2 x 3), exactly as notebook 02 computes it."""
    m, ref = experiment.video_mapping(json.loads((Path(results_dir) / clip / "clicks.json").read_text()))
    if m is None:
        raise ValueError(f"{clip}: no ruler (reference points: {ref})")
    return np.asarray(m["px_to_mm"], float)


def lab_table(sheet: Path = LAB_SHEET, sheet_name: str = LAB_SHEET_NAME) -> pd.DataFrame:
    """The lab's clip means, SDs, piece counts and standard errors (extruder's `labels`)."""
    return labels.clip_labels(labels.load_lab_pieces(Path(sheet), sheet_name, LAB_COLUMNS))


def matched_clips(lab: pd.DataFrame, pieces: pd.DataFrame) -> tuple[list[str], list[str]]:
    """(clips with lab values and pieces, clips with only one of the two), both sorted."""
    a, b = set(lab["clip"].astype(str)), set(pieces["clip"].astype(str))
    return sorted(a & b), sorted(a ^ b)


def pathway1_means(pset: pd.DataFrame) -> dict[str, dict[str, float]]:
    """Pathway 1 (zero training): each clip's mean diameter (`width_mm`) and length."""
    g = pset.groupby("clip")
    return {"diameter": {c: float(v) for c, v in g["width_mm"].mean().items()},
            "length": {c: float(v) for c, v in g["length_mm"].mean().items()}}

"""The lab measurements (calipers + thread), one row per measured piece.

The spreadsheet has one row per piece. A clip is named like the video file ("901_1" for
901_1.MP4); its first three characters are the process setting ("901"). Any number of pieces
per clip works (5 now, 10 later).
"""
from pathlib import Path

import numpy as np
import pandas as pd

DEFAULT_COLUMNS = dict(clip="Sl. No.", length="Length of the Product (mm)",
                       diameter="Diameter of the Product (mm)", piece="Piece no.")


def load_lab_pieces(path, sheet="Oct 7 Final", columns=None):
    """One row per measured piece: clip, setting, piece, length_mm, diameter_mm (+ all original columns)."""
    cols = {**DEFAULT_COLUMNS, **(columns or {})}
    path = Path(path)
    df = pd.read_csv(path) if path.suffix.lower() == ".csv" else pd.read_excel(path, sheet_name=sheet)
    missing = [c for c in cols.values() if c not in df.columns]
    if missing:
        raise KeyError(f"Columns not found in {path.name}: {missing}. Available: {list(df.columns)}")
    out = df.copy()
    out.insert(0, "clip", df[cols["clip"]].astype(str).str.strip())
    out.insert(1, "setting", out["clip"].str.split("_").str[0])
    out.insert(2, "piece", df[cols["piece"]])
    out.insert(3, "length_mm", pd.to_numeric(df[cols["length"]], errors="coerce"))
    out.insert(4, "diameter_mm", pd.to_numeric(df[cols["diameter"]], errors="coerce"))
    return out[out["clip"].ne("") & out["clip"].ne("nan")].reset_index(drop=True)


def clip_labels(pieces):
    """Per clip: mean, SD, number of pieces and standard error of the mean (SD / sqrt(n))."""
    rows = []
    for clip, g in pieces.groupby("clip", sort=True):
        r = dict(clip=clip, setting=g["setting"].iloc[0])
        for q in ("diameter", "length"):
            v = g[f"{q}_mm"].dropna()
            r[f"lab_{q}_mean"] = float(v.mean()) if len(v) else np.nan
            r[f"lab_{q}_sd"] = float(v.std()) if len(v) > 1 else np.nan
            r[f"lab_{q}_n"] = int(len(v))
            r[f"lab_{q}_se"] = r[f"lab_{q}_sd"] / np.sqrt(len(v)) if len(v) > 1 else np.nan
        rows.append(r)
    return pd.DataFrame(rows)


def noise_floor(labels):
    """How precisely the lab's own clip means are known: the median standard error over clips."""
    return {q: float(labels[f"lab_{q}_se"].median()) for q in ("diameter", "length")}

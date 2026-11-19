"""Pathway 2 baselines on clip values: B0 and DL-0 under leave one setting out.

B0 predicts the mean of the training clips' lab values. DL-0 multiplies a clip's Pathway 1
value by k = mean(lab) / mean(Pathway 1) over the training clips (notebook 02's author's
blueprint). Every learned number comes from the training clips of its round only.
A clip's setting is the part of its name before "_" ("901_3" -> "901").
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np
from scipy import stats

from learned_pathway.splits import leave_one_condition_out

METHODS = ("B0", "DL-0")


@dataclass(frozen=True)
class Round:
    held_out: str
    test: list[str]
    train: list[str]


def setting_of(clip: str) -> str:
    return clip.split("_")[0]


def setting_folds(clips: Sequence[str]) -> list[Round]:
    """Leave one setting out (main scheme), built by learned_pathway.splits."""
    folds = leave_one_condition_out({"clip_id": c, "condition_id": setting_of(c)} for c in clips)
    return [Round(f.held_out_condition, list(f.test_clip_ids), list(f.train_clip_ids)) for f in folds]


def clip_folds(clips: Sequence[str]) -> list[Round]:
    """Leave one clip out (extra scheme): an easier question, a new clip of a setting already seen."""
    return [Round(c, [c], sorted(x for x in clips if x != c)) for c in sorted(clips)]


def dl0_factor(lab: Sequence[float], video: Sequence[float]) -> float:
    """k = mean(lab) / mean(video) over the training clips."""
    return float(np.mean(lab) / np.mean(video))


def cross_predict(video: Mapping[str, float], lab: Mapping[str, float], folds: Sequence[Round],
                  method: str) -> dict[str, float]:
    """Each test clip's prediction, from a model fitted on that round's training clips only."""
    if method not in METHODS:
        raise ValueError(f"unknown method {method!r}; expected one of {METHODS}")
    pred: dict[str, float] = {}
    for f in folds:
        lab_train = [lab[c] for c in f.train]
        if method == "B0":
            value = float(np.mean(lab_train))
            pred.update({c: value for c in f.test})
        else:
            k = dl0_factor(lab_train, [video[c] for c in f.train])
            pred.update({c: k * video[c] for c in f.test})
    return pred


def paired_mae_difference(err_a: Sequence[float], err_b: Sequence[float], n_boot: int = 2000,
                          seed: int = 0) -> dict[str, float]:
    """MAE(a) - MAE(b) over the same clips, its 95% bootstrap interval (clips resampled together)
    and the two-sided Wilcoxon signed-rank p of the paired absolute errors (nan if all equal)."""
    a, b = np.abs(np.asarray(err_a, float)), np.abs(np.asarray(err_b, float))
    if a.shape != b.shape or a.ndim != 1 or len(a) < 2:
        raise ValueError("need two equally long lists of at least two clip errors")
    d = a - b
    idx = np.random.default_rng(seed).integers(0, len(d), size=(n_boot, len(d)))
    boot = d[idx].mean(1)
    p = float(stats.wilcoxon(a, b).pvalue) if np.any(d != 0) else float("nan")
    return {"mae_difference": float(d.mean()), "ci_low": float(np.percentile(boot, 2.5)),
            "ci_high": float(np.percentile(boot, 97.5)), "wilcoxon_p": p}


# ---- the length rule (notebook 03) --------------------------------------------------
# A piece is the rope that came out between two knife passes, so length = rope speed / cut rate.
# The rope speed is learned as the training clips' mean of (lab length x cut rate); the cut rate
# comes from the knife speed of the run sheet. A yardstick for length, not a video method.

BLADES = 3


def cut_rate_hz(knife_rpm: float, blades: int = BLADES) -> float:
    """Knife passes per second over one die hole: blades x rpm / 60."""
    return blades * float(knife_rpm) / 60.0


def length_rule(lab_length: Mapping[str, float], rate_hz: Mapping[str, float],
                folds: Sequence[Round]) -> dict[str, float]:
    """Each test clip's length = (training clips' mean rope speed) / (its cut rate)."""
    pred: dict[str, float] = {}
    for f in folds:
        speed = float(np.mean([lab_length[c] * rate_hz[c] for c in f.train]))
        pred.update({c: speed / rate_hz[c] for c in f.test})
    return pred

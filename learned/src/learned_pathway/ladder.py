"""Notebook 03's comparison: one table of predictions, the error measures, paired wins and figures.

Every method's predictions become rows in one table (`rows`), in the column layout of extruder's
Experiment 1 (`extruder.experiment.compare`), so extruder's `metrics` (MAE with a 95% bootstrap
interval, bias, Pearson r, noise floor) applies unchanged. SEI rows are derived from the diameter
rows: SEI = (d / 4.28)^2 per clip mean (AGENTS rule 10; the lab's SEI uses the clip-mean diameter,.

A method **wins** against a yardstick when its MAE is lower and the paired bootstrap 95% interval of
the difference excludes 0 (`pathway2.paired_mae_difference`). Clips of
one setting are not independent, so the interval over 20 clips is optimistic; the per-setting
errors are shown next to it.
"""
from __future__ import annotations

from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import pandas as pd

from learned_pathway import nb02, pathway2  # nb02 puts the extruder package on the path
from extruder import experiment  # noqa: E402

METHOD_ORDER = ("B0", "length rule", "Pathway 1", "DL-0", "DL-1", "DL-2")
D_DIE_MM = 4.28
MAIN_SCHEME = "leave one setting out"
AGAINST = {"diameter": ("B0",), "SEI": ("B0",), "length": ("B0", "length rule")}
COLUMNS = ["scheme", "variant", "version", "key", "target", "clip", "setting", "video", "lab", "lab_sd",
           "lab_se", "video_sd", "error"]


def rows(scheme: str, method: str, version: str, key: str, target: str, pred: Mapping[str, float],
         lab: pd.DataFrame) -> pd.DataFrame:
    """One row per predicted clip: prediction (`video`), the lab's mean, SD and SE, and the error."""
    L = lab.set_index("clip")
    out = []
    for clip, value in pred.items():
        lm = float(L.loc[clip, f"lab_{target}_mean"])
        out.append(dict(scheme=scheme, variant=method, version=version, key=key, target=target, clip=clip,
                        setting=pathway2.setting_of(clip), video=float(value), lab=lm,
                        lab_sd=float(L.loc[clip, f"lab_{target}_sd"]), lab_se=float(L.loc[clip, f"lab_{target}_se"]),
                        video_sd=np.nan, error=float(value) - lm))
    return pd.DataFrame(out, columns=COLUMNS)


def sei_rows(diameter_rows: pd.DataFrame, d_die: float = D_DIE_MM) -> pd.DataFrame:
    """SEI rows from diameter rows: (d / d_die)^2 for the prediction and the lab's clip mean."""
    s = diameter_rows.copy()
    s["target"] = "SEI"
    s["video"] = (diameter_rows["video"] / d_die) ** 2
    s["lab"] = (diameter_rows["lab"] / d_die) ** 2
    s["lab_se"] = 2 * diameter_rows["lab"] * diameter_rows["lab_se"] / d_die ** 2
    s["lab_sd"] = np.nan
    s["error"] = s["video"] - s["lab"]
    return s


def metric_table(comp: pd.DataFrame) -> pd.DataFrame:
    """extruder's error measures per scheme and crop version, plus the mean |error| per setting."""
    out = []
    for (scheme, version), g in comp.groupby(["scheme", "version"], sort=False):
        m = experiment.metrics(g)
        per = (g.assign(ae=g["error"].abs())
               .pivot_table(index=["variant", "key", "target"], columns="setting", values="ae", aggfunc="mean"))
        per.columns = [f"MAE_{c}" for c in per.columns]
        m = m.merge(per.reset_index(), on=["variant", "key", "target"], how="left")
        out.append(m.assign(scheme=scheme, version=version))
    return pd.concat(out, ignore_index=True)


def paired_table(comp: pd.DataFrame, against: Mapping[str, Sequence[str]] = AGAINST) -> pd.DataFrame:
    """Each method against each yardstick named for its target, over the same clips."""
    out = []
    for (scheme, key, target), g in comp.groupby(["scheme", "key", "target"], sort=False):
        for yard in against.get(target, ()):
            y = g[g["variant"].eq(yard)].set_index("clip")["error"]
            if y.empty:
                continue
            for (method, version), m in g.groupby(["variant", "version"], sort=False):
                if method == yard:
                    continue
                e = m.set_index("clip")["error"]
                common = sorted(set(e.index) & set(y.index))
                r = pathway2.paired_mae_difference(e.loc[common].to_numpy(), y.loc[common].to_numpy())
                out.append(dict(scheme=scheme, version=version, key=key, target=target, method=method,
                                yardstick=yard, clips=len(common), mae_method=float(e.loc[common].abs().mean()),
                                mae_yardstick=float(y.loc[common].abs().mean()), **r,
                                wins=bool(r["ci_high"] < 0)))
    return pd.DataFrame(out)


# ------------------------------------------------------------------ figures
def _plt():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 8})
    return plt


def _label(variant: str, version: str) -> str:
    return variant if version == "none" else f"{variant} ({version})"


def plot_ladder(metrics: pd.DataFrame, path: Path, scheme: str = MAIN_SCHEME, key: str = "red") -> Path:
    """MAE with its 95% interval per method, one panel per target; dashed line = noise floor."""
    plt = _plt()
    m = metrics[metrics["scheme"].eq(scheme) & metrics["key"].eq(key)]
    targets = [t for t in ("diameter", "SEI", "length") if t in set(m["target"])]
    fig, axes = plt.subplots(1, len(targets), figsize=(3.3 * len(targets), 2.8), squeeze=False)
    for ax, t in zip(axes[0], targets):
        g = m[m["target"].eq(t)].copy()
        g["order"] = [METHOD_ORDER.index(v) if v in METHOD_ORDER else 99 for v in g["variant"]]
        g = g.sort_values(["order", "version"])
        x = np.arange(len(g))
        ax.bar(x, g["MAE"], color="0.75")
        ax.errorbar(x, g["MAE"], [g["MAE"] - g["MAE_CI_low"], g["MAE_CI_high"] - g["MAE"]], fmt="none",
                    ecolor="k", capsize=2, lw=0.8)
        ax.axhline(float(g["noise_floor"].median()), ls="--", color="tab:red", lw=0.8, label="noise floor")
        ax.set_xticks(x, [_label(v, s) for v, s in zip(g["variant"], g["version"])], rotation=60, ha="right")
        ax.set_title(f"{t} ({key.upper()}, {scheme})", fontsize=8)
        ax.set_ylabel("MAE over clips" + (" (mm)" if t != "SEI" else ""))
    axes[0][0].legend(frameon=False)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return Path(path)


def plot_pred_vs_lab(comp: pd.DataFrame, method: str, key: str, version: str, path: Path,
                     scheme: str = MAIN_SCHEME) -> Path:
    """Predicted against lab, one point per clip coloured by setting, one panel per target."""
    plt = _plt()
    g0 = comp[comp["scheme"].eq(scheme) & comp["variant"].eq(method) & comp["key"].eq(key)
              & comp["version"].eq(version)]
    targets = [t for t in ("diameter", "SEI", "length") if t in set(g0["target"])]
    fig, axes = plt.subplots(1, len(targets), figsize=(3.0 * len(targets), 2.9), squeeze=False)
    for ax, t in zip(axes[0], targets):
        g = g0[g0["target"].eq(t)]
        for s, gs in g.groupby("setting"):
            ax.errorbar(gs["lab"], gs["video"], xerr=gs["lab_se"], fmt="o", ms=4, lw=0.6,
                        color=experiment.SET_COLOURS.get(s), label=s)
        lo, hi = float(min(g["lab"].min(), g["video"].min())), float(max(g["lab"].max(), g["video"].max()))
        pad = 0.05 * (hi - lo or 1)
        ax.plot([lo - pad, hi + pad], [lo - pad, hi + pad], "k--", lw=0.6)
        ax.set_xlabel(f"lab {t}")
        ax.set_ylabel(f"predicted {t}")
        ax.set_title(f"{_label(method, version)}, {key.upper()}", fontsize=8)
    axes[0][0].legend(frameon=False, fontsize=7)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return Path(path)


def gallery(images: Sequence[np.ndarray], labels: Sequence[str], path: Path, title: str = "",
            ncols: int = 8) -> Path:
    """A grid of crops with a label under each (contact sheets, DL-2's most and least trusted pieces)."""
    plt = _plt()
    n = len(images)
    nrows = max(1, -(-n // ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(1.3 * ncols, 1.45 * nrows), squeeze=False)
    for ax in axes.ravel():
        ax.axis("off")
    for ax, img, lab in zip(axes.ravel(), images, labels):
        ax.imshow(img, cmap="gray" if np.asarray(img).ndim == 2 else None, vmin=0, vmax=255)
        ax.set_title(lab, fontsize=6)
    fig.suptitle(title, fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return Path(path)


def plot_training_curve(losses: Sequence[float], path: Path) -> Path:
    plt = _plt()
    fig, ax = plt.subplots(figsize=(3.3, 2.2))
    ax.plot(np.arange(1, len(losses) + 1), losses, lw=0.8)
    ax.set_xlabel("training step")
    ax.set_ylabel("training loss (z-scored)")
    ax.set_yscale("log")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return Path(path)


def setting_permutations(clips: Sequence[str]) -> list[dict[str, str]]:
    """The 24 orders of four settings' labels (identity first): clip i of setting s takes the label of
    clip i of setting pi(s), clips sorted within a setting. Settings must have equally many clips."""
    from itertools import permutations

    by = {}
    for c in sorted(clips):
        by.setdefault(pathway2.setting_of(c), []).append(c)
    sizes = {len(v) for v in by.values()}
    if len(sizes) != 1:
        raise ValueError(f"settings have different clip counts: { {s: len(v) for s, v in by.items()} }")
    names = sorted(by)
    return [{c: by[t][i] for s, t in zip(names, order) for i, c in enumerate(by[s])}
            for order in permutations(names)]


def permutation_p(observed_mae: float, permuted_maes: Sequence[float]) -> float:
    """Share of permutations (the identity included) whose MAE is at most the observed one."""
    m = np.asarray(permuted_maes, float)
    return float(np.mean(m <= observed_mae + 1e-12))

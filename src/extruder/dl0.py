"""DL-0: the geometric clip average times one learned number k (the simplest learning baseline).

Plain words
-----------
1. For every clip, the geometric measurement (experiment.py) gives a mean diameter and a mean
   length from the chosen pieces (every mask, good masks, or only the long ones; see
   experiment.length_selections).
2. On the training clips:   k = average(lab clip means) / average(video clip means).
   On a test clip:          prediction = k x video clip mean.
   One k for diameter and one for length, for RED and BLUE separately.
3. Splits (a clip is never used to fit the k that predicts it):
   - leave one setting out: fit on the 15 clips of 3 settings, predict the 5 clips of the 4th;
   - leave one clip out: fit on 19 clips, predict the 20th.
   "no training" = k = 1, i.e. the geometric value itself (Experiment 1).
4. Choosing the piece selection fairly: for each test fold, the selection is chosen with the
   training clips only (the same kind of split, inside them), then used on the test clips.
"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image
from scipy.stats import wilcoxon

from . import experiment

TARGETS = ("diameter", "length")
SPLITS = ("leave one setting out", "leave one clip out")
GEOMETRIC, DL0 = "geometric (k = 1)", "DL-0 (k x geometric)"
NO_TRAINING = "no training"
CHOSEN = "chosen on training clips"


# ------------------------------------------------------------------ clip averages per selection
def clip_means(pieces, selections, labels=None):
    """One row per selection, clip and mask type: pieces chosen, mean and SD of diameter and length."""
    rows = []
    for name, spec in selections.items():
        p = pieces[experiment.select_pieces(pieces, spec)]
        for (clip, key), g in pieces.groupby(["clip", "key"]):
            s = p[(p["clip"] == clip) & (p["key"] == key)]
            r = dict(selection=name, clip=clip, key=key, setting=g["setting"].iloc[0], n_pieces=len(s))
            for t, col in experiment.TARGETS.items():
                r[f"video_{t}_mean"] = float(s[col].mean()) if len(s) else np.nan
                r[f"video_{t}_sd"] = float(s[col].std()) if len(s) > 1 else np.nan
            rows.append(r)
    cm = pd.DataFrame(rows)
    return cm if labels is None else cm.merge(labels, on=["clip", "setting"], how="left")


# ------------------------------------------------------------------ the model
def fit_k(video, lab):
    """k = average(lab) / average(video) over the training clips (clips with a missing value skipped)."""
    ok = np.isfinite(video) & np.isfinite(lab)
    return float(lab[ok].mean() / video[ok].mean()) if ok.any() else np.nan


def folds(d, split):
    """(fold name, boolean test mask) for each fold of a split, over the rows of d (one row per clip)."""
    if split == "leave one setting out":
        return [(s, (d["setting"] == s).to_numpy()) for s in sorted(d["setting"].unique())]
    if split == "leave one clip out":
        return [(c, (d["clip"] == c).to_numpy()) for c in d["clip"]]
    raise ValueError(split)


def _clip_data(cm, selection, key, t):
    d = cm[(cm["selection"] == selection) & (cm["key"] == key) & cm[f"lab_{t}_mean"].notna()]
    return d.sort_values("clip").reset_index(drop=True)


def _cv_mae(d, t, split):
    """Mean absolute error of DL-0 over the folds of `split`, inside the clips of d (used to choose a selection)."""
    x, y, err = d[f"video_{t}_mean"].to_numpy(), d[f"lab_{t}_mean"].to_numpy(), []
    for _, test in folds(d, split):
        if test.all():
            continue
        err.extend(np.abs(fit_k(x[~test], y[~test]) * x[test] - y[test]))
    err = np.asarray(err, float)
    return float(np.nanmean(err)) if np.isfinite(err).any() else np.inf


def _row(method, split, selection, fold, r, t, k):
    return dict(method=method, split=split, selection=selection, fold=fold, key=r["key"], target=t,
                clip=r["clip"], setting=r["setting"], k=k, geometric=r[f"video_{t}_mean"],
                video=k * r[f"video_{t}_mean"], video_sd=k * r[f"video_{t}_sd"], lab=r[f"lab_{t}_mean"],
                lab_sd=r[f"lab_{t}_sd"], lab_se=r[f"lab_{t}_se"], n_used=r["n_pieces"])


def predict_all(cm, splits=SPLITS):
    """Predictions for every clip: geometric (k = 1) and DL-0, for every selection, split, mask type, target.

    Also adds the rows with selection = CHOSEN, where each test fold uses the selection that worked
    best on its own training clips. Returns (predictions, chosen) where `chosen` lists the
    selection picked in each fold.
    """
    rows, chosen = [], []
    selections = list(dict.fromkeys(cm["selection"]))
    for key in sorted(cm["key"].unique()):
        for t in TARGETS:
            data = {s: _clip_data(cm, s, key, t) for s in selections}
            for s, d in data.items():
                for _, r in d.iterrows():
                    rows.append(_row(GEOMETRIC, NO_TRAINING, s, "", r, t, 1.0))
                for split in splits:
                    x, y = d[f"video_{t}_mean"].to_numpy(), d[f"lab_{t}_mean"].to_numpy()
                    for fold, test in folds(d, split):
                        if test.all():
                            continue   # nothing left to train on (only one setting or one clip)
                        k = fit_k(x[~test], y[~test])
                        for _, r in d[test].iterrows():
                            rows.append(_row(DL0, split, s, fold, r, t, k))
            # the selection chosen with the training clips only
            base = data[selections[0]]
            for split in (NO_TRAINING,) + tuple(splits):
                # geometric has nothing to fit; its selection is chosen with the folds of the main split
                fold_split = split if split != NO_TRAINING else splits[0]
                for fold, test in folds(base, fold_split):
                    if test.all():
                        continue
                    train_clips = set(base.loc[~test, "clip"])
                    test_clips = set(base.loc[test, "clip"])
                    score = {}
                    for s, d in data.items():
                        tr = d[d["clip"].isin(train_clips)]
                        if split == NO_TRAINING:   # geometric: plain error on the training clips
                            score[s] = float(np.nanmean(np.abs(tr[f"video_{t}_mean"] - tr[f"lab_{t}_mean"])))
                        else:
                            score[s] = _cv_mae(tr, t, split)
                    best = min(selections, key=lambda s: (score[s] if np.isfinite(score[s]) else np.inf))
                    d = data[best]
                    tr, te = d[d["clip"].isin(train_clips)], d[d["clip"].isin(test_clips)]
                    if split == NO_TRAINING:
                        method, k = GEOMETRIC, 1.0
                    else:
                        method = DL0
                        k = fit_k(tr[f"video_{t}_mean"].to_numpy(), tr[f"lab_{t}_mean"].to_numpy())
                    for _, r in te.iterrows():
                        rows.append({**_row(method, split, CHOSEN, fold, r, t, k), "chosen": best})
                    chosen.append(dict(method=method, split=split, key=key, target=t, fold=fold, chosen=best,
                                       training_mae=score[best], k=k))
    pred = pd.DataFrame(rows)
    pred["error"] = pred["video"] - pred["lab"]
    return pred, pd.DataFrame(chosen)


BY = ("method", "split", "selection", "key", "target")


def metrics(pred):
    """MAE (+ 95 % interval), RMSE, MAPE, bias, limits of agreement, R², correlation, noise floor, SD error."""
    return experiment.metrics(pred, by=BY)


def versus_geometric(pred, n_boot=2000, seed=0):
    """Is DL-0 better than the geometric value itself, on the same clips?

    For each split, selection, mask type and target: the MAE difference (DL-0 minus geometric),
    its 95 % paired bootstrap interval, and the Wilcoxon signed-rank p-value. DL-0 counts as
    better only if its MAE is lower AND the interval is entirely below 0.
    """
    rng = np.random.default_rng(seed)
    # same selection; for CHOSEN, the geometric value whose selection was also chosen on training clips
    geo = pred[pred["method"] == GEOMETRIC].set_index(["selection", "key", "target", "clip"])["error"]
    rows = []
    for (split, s, key, t), g in pred[pred["method"] == DL0].groupby(["split", "selection", "key", "target"], sort=False):
        ref = geo.reindex(pd.MultiIndex.from_frame(g[["selection", "key", "target", "clip"]]))
        a, b = np.abs(g["error"].to_numpy()), np.abs(ref.to_numpy())
        ok = np.isfinite(a) & np.isfinite(b)
        a, b = a[ok], b[ok]
        if len(a) < 2:
            continue
        diff = a - b
        boot = rng.choice(diff, size=(n_boot, len(diff)), replace=True).mean(1)
        lo, hi = np.percentile(boot, [2.5, 97.5])
        p = float(wilcoxon(a, b).pvalue) if np.any(diff != 0) else 1.0
        rows.append(dict(split=split, selection=s, key=key, target=t, clips=len(a), MAE_DL0=a.mean(),
                         MAE_geometric=b.mean(), difference=diff.mean(), diff_CI_low=lo, diff_CI_high=hi,
                         wilcoxon_p=p, DL0_better=bool(diff.mean() < 0 and hi < 0)))
    return pd.DataFrame(rows)


# ------------------------------------------------------------------ figures
SPLIT_STYLE = {(GEOMETRIC, NO_TRAINING): ("0.45", "geometric, no training (k = 1)"),
               (DL0, "leave one setting out"): ("tab:blue", "DL-0, leave one setting out"),
               (DL0, "leave one clip out"): ("tab:orange", "DL-0, leave one clip out")}


def _ladder(selections, ref):
    """Selections along the x axis for one length reference: every mask, good masks, ≥0.5 ... longest."""
    names = ["every mask", "good masks"]
    names += [s for s in selections if s.endswith(f" {ref}") or (ref == "longest" and s == "longest only")]
    return names


def _tick(name):
    return {"every mask": "every\nmask", "good masks": "good\nmasks", "longest only": "1.0\n(longest\nonly)"}.get(
        name, name.split()[0].lstrip("≥"))


def plot_ablation(met, key, selections, path=None):
    """MAE per clip against the piece selection; solid = relative to the longest, dashed = to the 95th percentile."""
    fig, ax = plt.subplots(1, 2, figsize=(16, 5))
    main = _ladder(selections, "longest")
    for a, t in zip(ax, TARGETS):
        m = met[(met["key"] == key) & (met["target"] == t)]
        for (method, split), (col, label) in SPLIT_STYLE.items():
            mm = m[(m["method"] == method) & (m["split"] == split)].set_index("selection")
            if mm.empty:
                continue   # split not run
            for ref, ls in (("longest", "-"), ("p95", "--")):
                names = _ladder(selections, ref)
                if len(names) <= 2 and ref == "p95":
                    continue
                y = mm["MAE"].reindex(names).to_numpy()
                a.plot(np.arange(len(names)), y, ls, marker="o", color=col, ms=5,
                       label=label if ref == "longest" else None, alpha=1 if ref == "longest" else 0.6)
            ch = m[(m["method"] == method) & (m["split"] == split) & (m["selection"] == CHOSEN)]
            if len(ch):
                a.axhline(ch["MAE"].iloc[0], color=col, ls=":", lw=1.5)
        nf = m["noise_floor"].median()
        a.axhline(nf, color="green", ls="--", lw=2, label=f"lab noise floor {nf:.2f} mm")
        a.set_xticks(np.arange(len(main))); a.set_xticklabels([_tick(n) for n in main], fontsize=9)
        a.set(xlabel="pieces used: every mask / good masks / good masks at least this fraction of the longest",
              ylabel="mean absolute error per clip (mm)", title=f"{t.capitalize()} ({key.upper()} masks)")
        a.grid(alpha=0.3)
    ax[0].legend(fontsize=8)
    fig.suptitle("Which pieces to use? Solid: fraction of the longest mask. Dashed: fraction of the 95th-percentile "
                 "length. Dotted: selection chosen on training clips", fontsize=11)
    plt.tight_layout()
    if path is not None:
        fig.savefig(path, dpi=90, bbox_inches="tight")
    return fig


def plot_pieces_kept(cm, selections, path=None):
    """How many pieces per clip each selection keeps (median over clips, and the smallest clip)."""
    names = _ladder(selections, "longest")
    fig, ax = plt.subplots(1, 2, figsize=(14, 3.8), sharey=True)
    for a, key in zip(ax, ("red", "blue")):
        g = cm[cm["key"] == key].groupby("selection")["n_pieces"]
        med, low = g.median().reindex(names), g.min().reindex(names)
        x = np.arange(len(names))
        a.bar(x, med, color="tab:red" if key == "red" else "tab:blue", alpha=0.75, label="median over clips")
        a.plot(x, low, "k_", ms=18, mew=2, label="smallest clip")
        for xi, v in zip(x, med):
            a.text(xi, v, f"{v:.0f}", ha="center", va="bottom", fontsize=8)
        a.set_xticks(x); a.set_xticklabels([_tick(n) for n in names], fontsize=8)
        a.set(title=f"{key.upper()} masks: pieces per clip", ylabel="pieces (log scale)"); a.set_yscale("log")
        a.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:g}"))
        a.yaxis.set_minor_formatter(plt.NullFormatter())
        a.legend(fontsize=8); a.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    if path is not None:
        fig.savefig(path, dpi=90, bbox_inches="tight")
    return fig


def _mask_crop(f, pad=6):
    m = np.asarray(Image.open(f)) > 0
    ys, xs = np.nonzero(m)
    if not len(ys):
        return m
    return m[max(0, ys.min() - pad):ys.max() + pad + 1, max(0, xs.min() - pad):xs.max() + pad + 1]


def plot_length_idea(pieces, clip, key="blue", n=8, thresholds=(0.5, 0.7, 0.9), path=None):
    """One clip: the spread of mask lengths, and masks from the shortest-looking to the longest."""
    g = pieces[(pieces["clip"] == clip) & (pieces["key"] == key) & pieces["used"]].dropna(subset=["length_ratio"])
    g = g.sort_values("length_ratio")
    fig = plt.figure(figsize=(16, 6.4))
    a = fig.add_subplot(2, 1, 1)
    a.hist(g["length_mm"], bins=30, color="tab:blue" if key == "blue" else "tab:red", alpha=0.75)
    longest = g["length_mm"].max()
    a.axvline(longest, color="k", lw=2, label=f"longest {longest:.1f} mm")
    for t, c in zip(thresholds, ("tab:orange", "tab:green", "tab:purple")):
        a.axvline(t * longest, color=c, ls="--", lw=1.5,
                  label=f"{t:.1f} × longest: {(g['length_ratio'] >= t).sum()} of {len(g)} pieces kept")
    a.set(xlabel="measured length (mm)", ylabel="pieces",
          title=f"Clip {clip}, good {key.upper()} masks: a piece pointing at the camera looks short, never long")
    a.legend(fontsize=8); a.grid(alpha=0.3)
    pick = g.iloc[np.unique(np.linspace(0, len(g) - 1, min(n, len(g))).round().astype(int))]
    crops = [_mask_crop(f) for f in pick["mask_file"]]
    H, W = max(c.shape[0] for c in crops), max(c.shape[1] for c in crops)
    for i, ((_, r), c) in enumerate(zip(pick.iterrows(), crops)):
        # same pixel scale in every panel, so the sizes can be compared by eye
        canvas = np.zeros((H, W), bool)
        y0, x0 = (H - c.shape[0]) // 2, (W - c.shape[1]) // 2
        canvas[y0:y0 + c.shape[0], x0:x0 + c.shape[1]] = c
        b = fig.add_subplot(2, len(pick), len(pick) + i + 1)
        b.imshow(canvas, cmap="gray")
        b.set_title(f"{r['length_mm']:.1f} mm\n{r['length_ratio']:.2f} × longest", fontsize=9)
        b.axis("off")
    plt.tight_layout()
    if path is not None:
        fig.savefig(path, dpi=90, bbox_inches="tight")
    return fig

"""Build notebook 03, the learned pathway (DL-0, DL-1, DL-2), with its outputs cleared.

    python learned/make_notebook.py [--target notebooks/03_learned_pathway.ipynb]

The notebook only calls and explains tested code in `learned/src/learned_pathway` (nb02, pathway2,
dl1, piece_features, dl2, ladder); it starts where notebook 02 stops. Edit this file, not the
notebook, and run it again: a test checks that the committed notebook is this build.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import nbformat

ROOT = Path(__file__).resolve().parents[1]
TARGET = ROOT / "notebooks/03_learned_pathway.ipynb"

CELLS: list[tuple[str, str]] = [
    ("md", """# Notebook 03: learning the lab's numbers from the masks (DL-0, DL-1, DL-2)

**CIS 831 Deep Learning project, Kansas State University**

Notebook 02 measured every piece in millimetres with a ruler and **learned nothing** from the lab
(Experiment 1, "Pathway 1"). This notebook asks the next question: **can a model that learns from the
lab's measurements predict them better?**

| Method | What it learns from the lab |
|---|---|
| **B0** (yardstick) | nothing from the video: it predicts the average of the training clips |
| **length rule** (yardstick, length only) | the rope speed: length = rope speed ÷ cut rate |
| **Pathway 1** (yardstick) | nothing: notebook 02's measurements |
| **DL-0** | one correction factor for Pathway 1 |
| **DL-1** | a frozen image network (DINOv2) describes each piece; a small linear model maps the clip's description to its size |
| **DL-2** | the same descriptions; a **small trained network** learns how much to trust each piece of a clip, and the clip's size |

**The test is always a setting the model has never seen.** We hold out one of the four settings
(901–904), fit everything on the other three, predict the held-out one, and repeat for all four.
Every choice a method makes (its settings, its scaling) is also made on the three training settings
only. The lab measured whole clips (5 pieces each), so **the clip is the unit**: one prediction and
one error per clip.

**Decided before the lab values were used:** the main comparison is RED masks, outline crops, leave
one setting out. Everything else is secondary.

### How to run
1. Run notebook 02 first: this notebook reads its `results/` (one folder per video and `experiment1/`).
2. Once: `bash learned/setup_dinov2.sh` puts the DINOv2 code and weights into `checkpoints/`.
3. Kernel **Python (extruder-sam2)**. Run the cells in order. The first run encodes about 52,000
   crops per crop version (about 10 min on an Apple GPU for outlines) and saves them in
   `results/experiment2/features/`; later runs reuse them."""),

    ("md", """## Step 1: Settings
Everything you may want to change is in this cell.
* `KEYS`: which masks: `"red"` (the cut moment, main) and/or `"blue"` (just detached).
* `VERSIONS`: which crops: `"outline"` (the piece's shape, main) and/or `"colour"` (its pixels).
* `RUN_DL2`, `RUN_CLIP_DIAGNOSTIC`, `RUN_PERMUTATION`: switch the slower parts on or off."""),
    ("code", """import sys
from pathlib import Path
PROJECT_DIR = Path.cwd().parent if Path.cwd().name == "notebooks" else Path.cwd()
sys.path.insert(0, str(PROJECT_DIR / "learned" / "src"))

import numpy as np
import pandas as pd
from IPython.display import display, Image as ShowImage

from learned_pathway import nb02, pathway2, dl1, dl2, piece_features, ladder

RESULTS_DIR = nb02.RESULTS_DIR      # notebook 02's results folder (one folder per video + experiment1/)
VIDEOS_DIR = nb02.VIDEOS_DIR        # the videos (only needed for colour crops)
LAB_SHEET = nb02.LAB_SHEET          # the lab sheet, tab "Oct 7 Final"
KEYS = ("red", "blue")
VERSIONS = ("outline", "colour")
RUN_DL2 = True
RUN_CLIP_DIAGNOSTIC = True
RUN_PERMUTATION = False
DEVICE = "auto"                     # for DINOv2: "auto" = Apple GPU if present, else CPU

TARGETS = ("diameter", "length")
MAIN = "leave one setting out"
OUT = RESULTS_DIR / "experiment2"
FIG = OUT / "figures"
FIG.mkdir(parents=True, exist_ok=True)
pd.set_option("display.width", 200)"""),

    ("md", """## Step 2: Lab measurements and noise floor
Per clip: the mean, SD, number of pieces and **standard error** (SE = SD / √n) of the caliper diameter
and the thread length. The median SE is the **noise floor**: the lab's own 5 pieces know their clip's
mean only this precisely, so no method can be judged more finely than that."""),
    ("code", """lab = nb02.lab_table(LAB_SHEET)
floor = {t: float(lab[f"lab_{t}_se"].median()) for t in TARGETS}
display(lab.round(2))
print("noise floor (median SE):", {t: round(v, 2) for t, v in floor.items()}, "mm")"""),

    ("md", """## Step 3: Pieces from notebook 02
We use the masks notebook 02 **kept** and whose shape **passed its checks** (`used`). We also drop
**repeats**: when a cut's RED frame is the same as the previous cut's, SAM 2 followed the same piece
twice. Every method below uses exactly these pieces.

Look at the share of cuts kept: it differs a lot between settings. Where few cuts are kept, the kept
pieces may not be typical of their clip (they are the ones that left the hole quickly)."""),
    ("code", """pieces = nb02.load_pieces(RESULTS_DIR)
CLIPS, left_out = nb02.matched_clips(lab, pieces)
print(f"{len(CLIPS)} clips have both lab values and pieces; left out: {left_out or 'none'}")
sets = {key: nb02.piece_set(pieces, key, CLIPS) for key in KEYS}

cuts = pd.concat([pd.read_csv(RESULTS_DIR / c / "cuts.csv").assign(setting=pathway2.setting_of(c)) for c in CLIPS])
summary = cuts.groupby("setting").agg(cuts=("cut", "size"), kept=("keep", "sum"))
summary["kept (%)"] = (100 * summary["kept"] / summary["cuts"]).round(1)
summary["repeats dropped"] = pieces[pieces["repeat"] & pieces["key"].eq("red")].groupby("setting").size()
for key in KEYS:
    summary[f"{key} pieces used"] = sets[key].groupby("setting").size()
display(summary)"""),

    ("md", """## Step 4: Why this is hard
**Diameter.** Look at the four settings' lab diameters below, and at the last line, which compares
the spread of the three closest settings with the noise floor. If three settings lie closer together
than the noise floor and the fourth differs, then, when the fourth is held out, a model must learn
"different" from three settings that are the same size; nothing can learn that. This would be a limit
of the data (4 settings), not of the methods.

**Length.** A piece is the rope that came out between two knife passes, so
length ≈ rope speed ÷ cut rate. Where the rope speed is about the same between settings, this simple
rule is a strong yardstick: a learned model must beat it to show that the video adds something."""),
    ("code", """by_setting = (lab.assign(setting=lab["clip"].map(pathway2.setting_of))
              .groupby("setting")[["lab_diameter_mean", "lab_length_mean"]].mean())
by_setting["knife (rpm)"] = [nb02.KNIFE_RPM[s] for s in by_setting.index]
by_setting["cut rate (1/s)"] = [pathway2.cut_rate_hz(nb02.KNIFE_RPM[s]) for s in by_setting.index]
by_setting["rope speed (mm/s)"] = by_setting["lab_length_mean"] * by_setting["cut rate (1/s)"]
display(by_setting.round(2))
d = by_setting["lab_diameter_mean"].sort_values()
print(f"three closest settings span {d.iloc[2] - d.iloc[0]:.2f} mm; noise floor {floor['diameter']:.2f} mm")"""),

    ("md", """## Step 5: Yardsticks
* **B0**: the mean of the training clips' lab values.
* **length rule**: rope speed = the training clips' mean of (lab length × cut rate); prediction = rope
  speed ÷ the clip's cut rate (3 knife blades).
* **Pathway 1**: notebook 02's measurements, averaged over the pieces of Step 3 (zero training). Below,
  next to notebook 02's own clip means (which still include repeats)."""),
    ("code", """lab_y = {t: lab.set_index("clip").loc[CLIPS, f"lab_{t}_mean"].to_dict() for t in TARGETS}
rate = {c: pathway2.cut_rate_hz(nb02.KNIFE_RPM[pathway2.setting_of(c)]) for c in CLIPS}
SCHEMES = {MAIN: pathway2.setting_folds(CLIPS)}
if RUN_CLIP_DIAGNOSTIC:
    SCHEMES["leave one clip out"] = pathway2.clip_folds(CLIPS)
p1 = {key: nb02.pathway1_means(sets[key]) for key in KEYS}

parts, choices = [], []
for scheme, folds in SCHEMES.items():
    for key in KEYS:
        for t in TARGETS:
            parts.append(ladder.rows(scheme, "B0", "none", key, t, pathway2.cross_predict({}, lab_y[t], folds, "B0"), lab))
            parts.append(ladder.rows(scheme, "Pathway 1", "none", key, t, p1[key][t], lab))
        parts.append(ladder.rows(scheme, "length rule", "none", key, "length",
                                 pathway2.length_rule(lab_y["length"], rate, folds), lab))

nb02_table = pd.read_csv(RESULTS_DIR / "experiment1" / "clip_table.csv").query("key == 'red'").set_index("clip")
display(pd.DataFrame({"diameter, notebook 02": nb02_table["video_diameter_mean"],
                      "diameter, here": pd.Series(p1[KEYS[0]]["diameter"]),
                      "length, notebook 02": nb02_table["video_length_mean"],
                      "length, here": pd.Series(p1[KEYS[0]]["length"])}).loc[CLIPS].round(2))"""),

    ("md", """## Step 6: DL-0
**One correction factor.** Pathway 1's numbers are off by a fairly steady factor (the pieces are fresh and point toward the
camera). DL-0 learns one number per target from the training clips,
k = mean(lab) ÷ mean(Pathway 1), and predicts Pathway 1 × k."""),
    ("code", """for scheme, folds in SCHEMES.items():
    for key in KEYS:
        for t in TARGETS:
            pred = pathway2.cross_predict(p1[key][t], lab_y[t], folds, "DL-0")
            parts.append(ladder.rows(scheme, "DL-0", "none", key, t, pred, lab))
print("DL-0 done")"""),

    ("md", """## Step 7: Crops
For DL-1 and DL-2, every mask is redrawn as a 224 × 224 picture at exactly **4 px per mm** (with the
clip's ruler), centred on the piece. A 10 mm piece is always 40 px long, so the size in the picture
means the same thing in every clip. Below: two random pieces per setting."""),
    ("code", """rng = np.random.default_rng(0)
for version in VERSIONS:
    show = pd.concat([g.iloc[rng.choice(len(g), 2, replace=False)] for _, g in sets[KEYS[0]].groupby("setting")])
    imgs = piece_features.crops_of(show, RESULTS_DIR, version, VIDEOS_DIR)
    path = ladder.gallery(imgs, [f"{r.clip} cut {r.cut}" for r in show.itertuples()],
                          FIG / f"crops_{version}.png", f"{version} crops, 4 px/mm (224 px = 56 mm)")
    display(ShowImage(filename=str(path)))"""),

    ("md", """## Step 8: DINOv2 features
**DINOv2** (Meta AI) is a large image network trained on millions of photos. We use the small version
(ViT-S/14) **frozen**: it is never trained here; it turns each crop into 768 numbers that describe
its shape. Each crop is also turned by 90°, 180° and 270° and mirrored (**8 views**), because a piece's
direction in the picture does not change its size; DL-2 trains on all views. The numbers are saved,
and re-used as long as the masks, rulers and crops are unchanged."""),
    ("code", """FEATS = {}
for version in VERSIONS:
    FEATS[version] = piece_features.build_features(pieces, RESULTS_DIR, version, OUT / "features",
                                                   videos_dir=VIDEOS_DIR, device=DEVICE)
    print(version, FEATS[version][1].shape, "(pieces, views, numbers)")"""),

    ("md", """## Step 9: DL-1
**Frozen features and a small linear model.** For each clip: the average and the spread (SD) of its pieces' 768 numbers. A linear model maps these to
the lab's value: the numbers are standardised, reduced to k summaries (PCA, k = 4, 8 or 16) and fitted
by ridge regression (strength α). k and α are chosen by leaving one **training** setting out at a
time; the held-out setting is never seen."""),
    ("code", """for version in VERSIONS:
    F = FEATS[version][1]
    for key in KEYS:
        B = piece_features.bags(sets[key], F)
        names, X = dl1.clip_features(np.concatenate([B[c][:, 0] for c in CLIPS]),
                                     np.repeat(CLIPS, [len(B[c]) for c in CLIPS]))
        Fc = dict(zip(names, X))
        for scheme, folds in SCHEMES.items():
            for t in TARGETS:
                pred, rec = dl1.cross_predict_dl1(Fc, lab_y[t], folds, "setting")
                parts.append(ladder.rows(scheme, "DL-1", version, key, t, pred, lab))
                choices += [dict(r, method="DL-1", version=version, key=key, target=t, scheme=scheme) for r in rec]
display(pd.DataFrame([c for c in choices if c["method"] == "DL-1" and c["scheme"] == MAIN]))"""),

    ("md", """## Step 10: DL-2
**A small trained attention network.** A clip is a **bag** of pieces. DL-2 (attention-based multiple-instance learning, Ilse et al. 2018):
1. reduces each piece's 768 numbers to 32 (PCA, fitted on the training pieces only);
2. a small layer turns them into 16 numbers per piece;
3. an **attention** layer gives each piece a trust score; the scores of a clip add up to 1;
4. the trust-weighted average of the pieces gives the clip's size.

It has 1,106 trained numbers. Training: every step sees all training clips, each with 64 random pieces
in random views. The strength of the weight penalty and the number of steps are chosen by leaving one
training setting out; 5 trainings with different random seeds are averaged. It runs on the CPU, and the
same seeds always give the same numbers.

Below: the training curve of the first round, and the pieces DL-2 trusts most and least."""),
    ("code", """ATT = {}
if RUN_DL2:
    for version in VERSIONS:
        for key in KEYS:
            B = piece_features.bags(sets[key], FEATS[version][1])
            for t in TARGETS:
                pred, rec, att = dl2.cross_predict_dl2(B, lab_y[t], SCHEMES[MAIN])
                parts.append(ladder.rows(MAIN, "DL-2", version, key, t, pred, lab))
                choices += [dict(r, method="DL-2", version=version, key=key, target=t, scheme=MAIN) for r in rec]
                ATT[(version, key, t)] = att
                print(f"DL-2 {version} {key} {t} done")
    display(pd.DataFrame([c for c in choices if c["method"] == "DL-2"]).drop(columns="digest"))"""),
    ("code", """if RUN_DL2:
    version, key, t = VERSIONS[0], KEYS[0], "diameter"
    B = piece_features.bags(sets[key], FEATS[version][1])
    rnd, rec = SCHEMES[MAIN][0], next(c for c in choices if c["method"] == "DL-2" and c["target"] == t)
    curve = dl2.training_curve({c: B[c] for c in rnd.train}, lab_y[t], dl2.Config(rec["weight_decay"], rec["epochs"]))
    display(ShowImage(filename=str(ladder.plot_training_curve(curve, FIG / "dl2_training_curve.png"))))
    pset = sets[key]
    picks = []
    for s, g in pset.groupby("setting"):
        clip = g["clip"].iloc[0]
        w = ATT[(version, key, t)][clip]
        order = np.argsort(w)
        rows = g[g["clip"].eq(clip)].iloc[np.r_[order[-4:][::-1], order[:4]]]
        picks += [(r, f"{clip} w={w[i]:.4f}") for (_, r), i in zip(rows.iterrows(), np.r_[order[-4:][::-1], order[:4]])]
    imgs = piece_features.crops_of(pd.DataFrame([p[0] for p in picks]), RESULTS_DIR, version, VIDEOS_DIR)
    path = ladder.gallery(imgs, [p[1] for p in picks], FIG / "dl2_attention.png",
                          "DL-2 diameter: per setting, 4 most trusted (left) and 4 least trusted pieces")
    display(ShowImage(filename=str(path)))"""),

    ("md", """## Step 11: Results
**MAE** = the average size of the error per clip (95% bootstrap interval in brackets); **bias** = the
average signed error; **MAE_901 … MAE_904** = the error per held-out setting. SEI = (d / 4.28)² is
computed from each method's diameter.

**Does a method win?** Only if its MAE is lower than the yardstick's *and* the 95% interval of the
paired difference stays below 0 (`wins`). Yardsticks: B0 for diameter and SEI; B0 and the length rule
for length.

**Caveat.** The five clips of one setting are not independent, so an interval over 20 clips is
optimistic; look at the per-setting errors too."""),
    ("code", """comp = pd.concat(parts, ignore_index=True)
comp = pd.concat([comp, ladder.sei_rows(comp[comp["target"].eq("diameter")])], ignore_index=True)
met = ladder.metric_table(comp)
paired = ladder.paired_table(comp)
cols = ["variant", "version", "key", "target", "MAE", "MAE_CI_low", "MAE_CI_high", "bias", "correlation", "noise_floor"]
cols += [c for c in met.columns if c.startswith("MAE_9")]
for key in KEYS:
    print(f"==== {key.upper()} masks, {MAIN} ====")
    display(met[met["scheme"].eq(MAIN) & met["key"].eq(key)][cols].round(3))
display(paired[paired["scheme"].eq(MAIN)].drop(columns="scheme").round(3))"""),
    ("code", """for key in KEYS:
    display(ShowImage(filename=str(ladder.plot_ladder(met, FIG / f"ladder_{key}.png", key=key))))
best = "DL-2" if RUN_DL2 else "DL-1"
for version in VERSIONS:
    path = ladder.plot_pred_vs_lab(comp, best, KEYS[0], version, FIG / f"pred_vs_lab_{best}_{version}.png")
    display(ShowImage(filename=str(path)))"""),
    ("md", """**Permutation check (optional, `RUN_PERMUTATION`).** How often would DL-1 do this well by chance?
We give each setting's five lab values to another setting (clip 1 to clip 1, and so on), in all
24 orders of the four settings, and rerun DL-1 (first mask type and crop version, leave one setting
out) on each. **p** is the share of the 24 orders, the true one included, with an MAE at most the
true one; the smallest possible p is 1/24 ≈ 0.04."""),
    ("code", """if RUN_PERMUTATION:
    B = piece_features.bags(sets[KEYS[0]], FEATS[VERSIONS[0]][1])
    names, X = dl1.clip_features(np.concatenate([B[c][:, 0] for c in CLIPS]),
                                 np.repeat(CLIPS, [len(B[c]) for c in CLIPS]))
    Fc, perm = dict(zip(names, X)), []
    for t in TARGETS:
        for i, m in enumerate(ladder.setting_permutations(CLIPS)):
            yp = {c: lab_y[t][m[c]] for c in CLIPS}
            pred, _ = dl1.cross_predict_dl1(Fc, yp, SCHEMES[MAIN], "setting")
            perm.append(dict(target=t, order=i, MAE=float(np.mean([abs(pred[c] - yp[c]) for c in CLIPS]))))
    perm = pd.DataFrame(perm)
    perm.to_csv(OUT / "permutation.csv", index=False)
    display(pd.DataFrame([dict(target=t, MAE_true=g["MAE"].iloc[0], MAE_median_of_24=g["MAE"].median(),
                               p=ladder.permutation_p(g["MAE"].iloc[0], g["MAE"])) for t, g in perm.groupby("target")]))"""),

    ("md", """## Step 12: Leave one clip out (diagnostic)
Here the held-out clip's own setting stays in training: four sister clips of the same setting are
seen. This is a much easier question, and a method that simply **recognises the setting** does well
here without reading size. That is why it is not the main result. (DL-2 is left out: 20 rounds would
take about five times as long.)"""),
    ("code", """if RUN_CLIP_DIAGNOSTIC:
    display(met[met["scheme"].eq("leave one clip out")][cols].round(3))"""),

    ("md", """## Step 13: Save everything
| File | Content |
|---|---|
| `experiment2/predictions.csv` | one row per scheme, method, crop version, mask type, target and clip |
| `experiment2/metrics.csv` | MAE (+ 95% interval), RMSE, bias, r, noise floor, MAE per setting |
| `experiment2/paired.csv` | each method against each yardstick: MAE difference, interval, `wins` |
| `experiment2/choices.csv` | the settings each learned method chose in each round |
| `experiment2/pieces_used.csv` | the pieces every method used |"""),
    ("code", """comp.to_csv(OUT / "predictions.csv", index=False)
met.to_csv(OUT / "metrics.csv", index=False)
paired.to_csv(OUT / "paired.csv", index=False)
pd.DataFrame(choices).to_csv(OUT / "choices.csv", index=False)
pd.concat(sets.values()).to_csv(OUT / "pieces_used.csv", index=False)
print("saved to", OUT)"""),

    ("md", """## Notes and limits
* **Four settings.** Every learned method trains on three. With so few settings, the clip-to-clip
  differences a model can learn are small, and diameter differs between settings by less than the lab's
  noise. More settings, and more random pieces per clip, are what would make learning worthwhile.
* **What the video sees.** The masks show fresh pieces at the die, pointing toward the camera; the lab
  measured cooled pieces days later, from all three holes, chosen by eye. A learned model maps one to
  the other; it does not make them the same thing.
* **Kept pieces.** Where few cuts are kept (Step 3), the kept pieces may not be typical of their clip.
* **Rulers.** Millimetres come from the per-video bolt clicks of notebook 02; the scatter of the
  clicks adds a little scale noise to each clip."""),
]


def build() -> nbformat.NotebookNode:
    nb = nbformat.v4.new_notebook()
    nb.metadata["kernelspec"] = {"name": "extruder-sam2", "display_name": "Python (extruder-sam2)",
                                 "language": "python"}
    nb.cells = [nbformat.v4.new_markdown_cell(src) if kind == "md" else nbformat.v4.new_code_cell(src)
                for kind, src in CELLS]
    for i, c in enumerate(nb.cells):
        c.id = f"cell-{i:02d}"  # stable ids, so a rebuilt notebook is byte-identical
    return nb


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--target", default=str(TARGET))
    args = ap.parse_args(argv)
    nb = build()
    nbformat.write(nb, args.target)
    print("wrote", args.target)


if __name__ == "__main__":
    main()

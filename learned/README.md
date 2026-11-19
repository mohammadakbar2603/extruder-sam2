# learned/: notebook 03, the learned pathway

Notebook 02 measures every piece in millimetres and learns nothing from the lab (Experiment 1,
"Pathway 1"). `notebooks/03_learned_pathway.ipynb` asks the next question: **can a model that learns
from the lab's measurements predict them better, for a setting it has never seen?**

This folder adds that notebook and its code. **It changes none of the existing files**: it reads
notebook 02's `results/` and uses the `extruder` package as it is.

| Method | What it learns from the lab |
|---|---|
| **B0** (yardstick) | nothing from the video: the average of the training clips |
| **length rule** (yardstick, length only) | the rope speed: length = rope speed ÷ cut rate (3 blades × knife rpm / 60) |
| **Pathway 1** (yardstick) | nothing: notebook 02's measurements |
| **DL-0** | one correction factor for Pathway 1 |
| **DL-1** | frozen DINOv2 features of each piece; the clip's mean and spread → PCA → ridge regression |
| **DL-2** | the same features; a small attention network (1,106 weights) learns how much to trust each piece |

The test is always a setting the model has never seen: one of the four settings is held out, everything
(including every setting of a method) is fitted on the other three, and this is repeated for all four.
The clip is the unit: one prediction and one error per clip.

## Setup (once, after `setup_env.sh`)

    bash learned/setup_dinov2.sh          # DINOv2 code (fixed commit) + weights into checkpoints/
    conda run -n extruder-sam2 pip install pytest nbformat   # only for the tests and the notebook builder

## Running it

1. Run notebook 02 (it writes `results/<video>/` and `results/experiment1/`).
2. Open `notebooks/03_learned_pathway.ipynb`, choose the kernel **Python (extruder-sam2)**, run the cells
   in order. Step 1 has the settings (mask type, crop version, which parts to run).
3. The first run describes every piece with DINOv2 (about 10 min for the outline crops on an Apple GPU;
   colour crops also read the videos) and saves the result in `results/experiment2/features/`. Later
   runs reuse it as long as the masks, rulers and crops are unchanged.

## What you get (`results/experiment2/`)

| File | Content |
|---|---|
| `predictions.csv` | one row per scheme, method, crop version, mask type, target and clip |
| `metrics.csv` | MAE (+ 95% interval), RMSE, bias, r, noise floor, MAE per held-out setting |
| `paired.csv` | each method against each yardstick: MAE difference, its 95% interval, `wins` |
| `choices.csv` | the settings each learned method chose in each round |
| `pieces_used.csv` | the pieces every method used |
| `permutation.csv` | only with `RUN_PERMUTATION = True` |
| `figures/` | crops, the DL-2 training curve and attention, the method ladder, predicted against lab |

## Code layout

| File | What it does |
|---|---|
| `src/learned_pathway/nb02.py` | reads notebook 02's results: pieces, repeats, rulers, lab clip means |
| `src/learned_pathway/splits.py` | leave one setting out, with inner folds |
| `src/learned_pathway/pathway2.py` | B0, DL-0, the length rule, the paired bootstrap |
| `src/learned_pathway/dl1.py`, `dl1_embed.py` | crops at 4 px/mm, DINOv2 input/output, DL-1 |
| `src/learned_pathway/piece_features.py` | outline/colour crops in 8 views → DINOv2 features, cached |
| `src/learned_pathway/dl2.py` | DL-2: gated-attention multiple-instance learning (Ilse et al. 2018) |
| `src/learned_pathway/ladder.py` | the comparison tables (with `extruder.experiment.metrics`) and figures |
| `make_notebook.py` | writes `notebooks/03_learned_pathway.ipynb` (edit this, not the notebook) |
| `tests/` | tests on SYNTHETIC data: `python -m pytest learned/tests` |

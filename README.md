# extruder-sam2

**Finding and measuring each extruded corn puff at the moment the knife cuts it, from slow-motion video, with SAM 2.1.**

CIS 831 Deep Learning, Kansas State University (Fall 2026).
Team: Mir Tuhin Billah (Grain Science), Mohammad Akbar Bin Shah (Computer Science), Sanaz Gheibuni (Computer Science).
Mentors: Dr. Sajid Alavi, Dr. Doina Caragea.

---

## What it does

A camera records the die face of a Wenger X20 single-screw extruder at about 240 frames per second. For **every knife cut at die hole A**, the notebook:

1. **Finds the cut.** A small spot on hole A "blinks" (dark when the knife passes, yellow when a new rope end comes out). The rhythm of the blink gives the cuts per second, so the knife speed is not needed.
2. **Points SAM 2 at the new rope end automatically.** On the frame right after each blink, a colour rule finds the yellow blob on the hole and draws a box around it. SAM 2 receives this box as if a person had drawn it.
3. **Lets SAM 2 follow the piece** for the next 16 frames.
4. **Marks two frames:**
   - **RED** is the last frame where the piece still touches hole A (the cut moment).
   - **BLUE** is the first frame where it has come free.
5. **Saves both masks** with the cut number and frame number in the name (`cut0007_red_f01762.png`, `cut0007_blue_f01763.png`) and **measures** both pieces: width, length, area and expansion.
6. **Converts pixels to millimetres** using the two bolts on the die plate as a ruler (their spacing is known: 65 mm).
7. **Summarises the video** as mean ± SD over all kept cuts.

The only manual step is **clicking hole A and the two bolts once per video** in an interactive picture.

| Step | What you see in the notebook |
|---|---|
| Crop | full frame → 360 × 360 px crop around hole A → close-up |
| Cuts | the blink signal, its rhythm, one full knife cycle |
| Prompt | head-on frame → yellowness map → yellow blob + box → SAM 2 mask |
| Tracking | green (attached) → **RED** (cut) → **BLUE** (detached), as a picture strip and an animation |
| Results | gallery of random red/blue masks, histograms, mean ± SD table |

On our test clip (10.7 s), 244 cuts were found and 206 were kept by the automatic checks. Processing took about 6 minutes on an RTX 5080.

## Setup (once per computer)

You need an **NVIDIA GPU**, **conda** (Miniconda or Anaconda), **git** and **VS Code** with the *Python* and *Jupyter* extensions.

### Linux / macOS / WSL: one command
```bash
git clone https://github.com/mohammadakbar2603/extruder-sam2.git
cd extruder-sam2
bash setup_env.sh
```
The script:
- creates the conda environment `extruder-sam2` (Python 3.11, PyTorch for CUDA 12.8, ffmpeg and the other packages);
- installs SAM 2.1 (a fixed version, so everyone runs the same code);
- downloads the SAM 2.1-large weights (about 900 MB) into `checkpoints/`;
- registers the notebook kernel **Python (extruder-sam2)**.

### Windows: the same steps by hand (Anaconda Prompt)
```bat
conda env create -f environment.yml
conda activate extruder-sam2
set SAM2_BUILD_CUDA=0
pip install --no-build-isolation "git+https://github.com/facebookresearch/sam2.git@2b90b9f5ceec907a1c18123530e92e794ad901a4"
pip install -e .
python -m ipykernel install --user --name extruder-sam2 --display-name "Python (extruder-sam2)"
curl -L -o checkpoints\sam2.1_hiera_large.pt https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_large.pt
```

## Running it

1. Copy your videos into `data/`. Videos are not stored in this repository.
2. Open `notebooks/01_cut_frame_pipeline.ipynb` in VS Code and choose the kernel **Python (extruder-sam2)**.
3. In **Step 2 (Configuration)**, set `VIDEO` to your file name.
4. Run the cells **one at a time**. At **Step 4**:
   - click hole A, the bolt just below it and the other bolt;
   - press **Save points**;
   - continue.
5. Results appear in `results/<video name>/`.

To process many videos: click and save points for each one, then use Step 12 (`RUN_ALL = True`).

## What you get (`results/<video name>/`)

| File | Content |
|---|---|
| `cuts.csv` | one row per cut: event, head-on, RED and BLUE frame numbers, kept or rejected (with the reason), and all measurements for RED (`red_*`) and BLUE (`blue_*`) |
| `summary.json`, `summary.csv` | mean and SD of every measurement over the kept cuts |
| `masks/` | black-and-white masks, e.g. `cut0007_red_f01762.png` (360 × 360 crop coordinates) |
| `overlays/` | the same masks drawn on the image, to check by eye |
| `figures/` | every figure from the notebook, plus `example_tracking.gif` |
| `clicks.json` | your clicks, the refined hole A centre, and the scale in px/mm |
| `crop_frames/` | the 360 × 360 crop of every frame (working files; can be deleted afterwards) |

### Measurements

| Name | Meaning |
|---|---|
| `width_mm` | typical thickness of the piece (median along its middle line, ends excluded). The most reliable number |
| `width_max_mm`, `width_mean_mm` | thickest point; area ÷ length |
| `length_mm` | tip to tip along the middle line, as seen on the die face |
| `area_mm2` | area covered by the piece in the picture |
| `feret_mm` | longest straight distance across the piece |
| `straightness` | 1 = straight, smaller = more curved |
| `expansion` | (width ÷ 4.28 mm die hole)². This is the apparent sectional expansion: hot, at the die, seen from one side |

Every measurement is also given in pixels (`*_px`), so results stay usable even without the bolt clicks.

## Pixels to millimetres

The lab measured the cold die plate:

| Feature | Value |
|---|---|
| Die holes | 4.28 mm, on a 31 mm circle |
| Bolts | 14 mm heads, on a 45 mm circle |
| Hole A to hole B | 55 mm |
| Hole A to the near bolt | 16 mm |
| Bolt to bolt | 65 mm |

`src/extruder/calib.py` turns these into a millimetre map of the plate and matches it to the clicked bolts.

- **Only the bolts set the scale by default.** In product videos the holes are always covered, and the rope sticks out toward the angled camera, so the visible hole is shifted by about 2 mm.
- **Limitation:** with two bolts, the mapping uses one scale for the whole image. On our blank-die video the camera angle stretches one direction by about ±3 %.
- **To correct the stretch:** record 2–3 s of the bare die whenever the camera is mounted, so all four points (both holes and both bolts) can be used.

## Code layout

```
notebooks/01_cut_frame_pipeline.ipynb   the step-by-step notebook (start here)
src/extruder/
  video.py     reading the 10-bit HEVC videos with ffmpeg, the "clean die" median picture
  cuts.py      blink signal, cut rhythm, cut events, exact hole A centre
  prompt.py    the automatic box prompt
  track.py     SAM 2 tracking per cut, green / RED / BLUE decision
  calib.py     die-plate map in mm, pixel -> mm mapping
  measure.py   measurements from a mask
  pipeline.py  the steps glued together, quality checks, saving
  viz.py       all figures
  ui.py        the click tool
environment.yml, setup_env.sh, pyproject.toml
data/          your videos (not in git)
checkpoints/   SAM 2 weights (downloaded by setup)
results/       outputs (not in git)
```

## Troubleshooting

- **The click picture does not appear in VS Code.**
  - Make sure the *Jupyter* extension is installed and the kernel is **Python (extruder-sam2)**, then restart the kernel.
  - As a fallback, type the points in `MANUAL_POINTS` in Step 2, e.g. `{"hole_A": (723, 487), "bolt_near": (690, 533), "bolt_far": (854, 379)}`. You can read pixel positions by hovering over any plotted frame.
- **Very few cuts found.** Check the blink figure in Step 6. Hole A may be wrong (re-click it), or the video may be too dark or steamy.
- **Out of GPU memory.** Close other notebooks. Each cut uses only 17 frames, so memory use is small.

## Credits

- [SAM 2](https://github.com/facebookresearch/sam2) by Meta AI (Apache 2.0). We use the SAM 2.1 Hiera-Large checkpoint without retraining.

"""SAM 2 tracking of one piece per cut, and the green / red / blue decision.

For every cut we start a *fresh* SAM 2 video session (so pieces never get mixed up):

* the session holds the head-on frame (event + 1) and the next `n_track` frames;
* the automatic box (see prompt.py) is given on the first of these frames;
* SAM 2 follows the piece forward and returns one mask per frame.

Then, on every frame, we ask one question: does the mask overlap a small square
(16 x 16 px) centred on hole A?

* GREEN  = yes: the piece is still attached to the die;
* BLUE   = the first frame where the mask has left the square: the piece is cut free;
* RED    = the last green frame before the first blue one: the moment of the cut.

Anything after the first blue frame is ignored (SAM 2 sometimes jumps onto the next
rope end after the cut; that does not matter because we stop at the first blue).
"""
import os
import shutil
import tempfile
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import torch


def quiet_sam2():
    """Hide SAM 2's own progress bars (one per cut would flood the notebook) and one harmless warning.

    The warning says SAM 2's optional compiled extension is missing, so it skips filling
    holes smaller than 8 px inside masks. Our measurements are not affected in practice.
    """
    import warnings
    warnings.filterwarnings("ignore", message=".*post-processing.*")
    warnings.filterwarnings("ignore", message=".*cannot import name '_C'.*")
    import sam2.sam2_video_predictor as vp
    import sam2.utils.misc as misc
    for mod in (vp, misc):
        if hasattr(mod, "tqdm"):
            mod.tqdm = lambda it, *a, **k: it


def autocast_for(device):
    """Use bfloat16 on GPUs that support it (faster, same results in practice)."""
    if device == "cuda" and torch.cuda.is_bf16_supported():
        return torch.autocast("cuda", dtype=torch.bfloat16)
    return nullcontext()


def touches_hole(mask, hole_xy, r=8):
    """True if the mask overlaps the (2r x 2r) square centred on the hole."""
    hx, hy = int(round(hole_xy[0])), int(round(hole_xy[1]))
    return bool(mask[hy - r:hy + r, hx - r:hx + r].any())


def _window_folder(frame_files, start, stop, folder):
    """SAM 2 reads a folder of JPEGs named 00000.jpg, 00001.jpg, ... Make one for this cut."""
    shutil.rmtree(folder, ignore_errors=True)
    os.makedirs(folder)
    for i, src in enumerate(frame_files[start:stop]):
        dst = os.path.join(folder, f"{i:05d}.jpg")
        try:
            os.link(src, dst)  # no copy needed when possible
        except OSError:
            shutil.copyfile(src, dst)
    return folder


def track_cut(predictor, frame_files, f0, box, hole_xy, device="cuda", n_track=16, r=8, min_area=50,
              work_dir=None):
    """Track the piece that starts at frame f0 (head-on frame) and find red / blue frames.

    frame_files : list of all crop JPEG paths of the video (index = video frame number)
    box         : automatic box on frame f0, crop coordinates [x0, y0, x1, y1]
    Returns dict with masks {frame: bool array}, timeline string, red_frame, blue_frame.
    """
    stop = min(len(frame_files), f0 + n_track + 1)
    tmp = work_dir or tempfile.mkdtemp(prefix="sam2_cut_")
    folder = _window_folder(frame_files, f0, stop, os.path.join(tmp, "window"))
    masks = {}
    with torch.inference_mode(), autocast_for(device):
        state = predictor.init_state(video_path=folder)
        predictor.add_new_points_or_box(state, frame_idx=0, obj_id=1, box=np.asarray(box, np.float32))
        for fidx, obj_ids, logits in predictor.propagate_in_video(state, start_frame_idx=0,
                                                                  max_frame_num_to_track=n_track):
            masks[f0 + fidx] = (logits[0, 0] > 0).float().cpu().numpy().astype(bool)
        predictor.reset_state(state)
    del state
    if work_dir is None:
        shutil.rmtree(tmp, ignore_errors=True)

    frames = sorted(masks)
    status = {}
    for f in frames:
        if touches_hole(masks[f], hole_xy, r):
            status[f] = "A"  # attached (green)
        elif masks[f].sum() > min_area:
            status[f] = "f"  # free (blue)
        else:
            status[f] = "."  # lost / empty
    timeline = "".join(status[f] for f in frames)
    blue = next((f for f in frames if status[f] == "f"), None)
    red = None
    if blue is not None:
        attached_before = [f for f in frames if f < blue and status[f] == "A"]
        red = attached_before[-1] if attached_before else None
    return dict(masks=masks, status=status, timeline=timeline, red_frame=red, blue_frame=blue)

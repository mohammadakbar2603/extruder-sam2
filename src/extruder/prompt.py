"""The automatic box prompt: how the computer "points" SAM 2 at the new rope end.

SAM 2 needs to be shown the object once (a click or a box). Instead of a person, the
computer draws the box, on the head-on frame (event + 1) of every cut:

1. make the yellowness map (red minus blue) and smooth it a little;
2. threshold it halfway between the dark die and the brightest yellow near the hole
   (looking only inside an 80 x 80 px window around the hole);
3. keep the yellow blob that sits on the hole;
4. draw a box around it, limited to +/-35 px around the hole so that a big
   neighbouring piece cannot stretch it, plus a 3 px margin.

This is exactly the method used in the first test (cut-frame figure), unchanged.
All coordinates here are in the 360 x 360 crop, where hole A is at (HOLE, HOLE).
"""
import cv2
import numpy as np

from .cuts import yellowness


def disc_box(img, hole_xy, near=14, local=40, limit=35, margin=3, return_steps=False):
    """Box [x0, y0, x1, y1] around the fresh rope end at the hole, or None if no blob.

    img      : RGB crop (H, W, 3)
    hole_xy  : hole A centre in crop pixels
    near     : the blob must overlap a (2*near) square centred on the hole
    local    : threshold is computed from a (2*local) window around the hole
    limit    : the box may not reach further than this from the hole
    """
    hx, hy = int(round(hole_xy[0])), int(round(hole_xy[1]))
    y = cv2.GaussianBlur(yellowness(img), (0, 0), 1.2)
    c = y[hy - local:hy + local, hx - local:hx + local]
    thr = 0.5 * (np.percentile(c, 10) + np.percentile(c, 99))
    n, lab = cv2.connectedComponents((y > thr).astype(np.uint8))
    reg = lab[hy - near:hy + near, hx - near:hx + near]
    ids = [k for k in np.unique(reg) if k > 0]
    steps = dict(yellow_map=y, threshold=thr, above_threshold=y > thr, blob=None, box=None)
    if not ids:
        return (None, steps) if return_steps else None
    k = max(ids, key=lambda k: (reg == k).sum())
    ys, xs = np.nonzero(lab == k)
    x0, x1 = max(xs.min(), hx - limit), min(xs.max(), hx + limit)
    y0, y1 = max(ys.min(), hy - limit), min(ys.max(), hy + limit)
    box = np.array([x0 - margin, y0 - margin, x1 + margin, y1 + margin], np.float32)
    steps.update(blob=lab == k, box=box)
    return (box, steps) if return_steps else box

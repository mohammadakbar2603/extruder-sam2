"""From pixels to millimetres, using known points on the die plate.

Plain words
-----------
The camera looks at the die plate at an angle and is re-mounted after cleaning, so
"how many pixels make one millimetre" changes a little from video to video, and
even across the image. We fix this with reference points whose true positions on
the die plate we know (measured on the cold die by the lab):

    die holes (A and B)   lie on a circle of radius 31 mm around the plate centre
    bolts (near and far)  lie on a circle of radius 45 mm around the plate centre
    hole A - hole B       55 mm
    hole A - near bolt    16 mm   (the bolt just below hole A)
    bolt   - bolt         65 mm
    die hole diameter     4.28 mm,   bolt head diameter 14 mm

From these distances we build a small "map" of the die plate in millimetres
(`die_map_mm`). You click hole A and the two bolts in the video (hole B optional);
the computer then finds the best mapping between video pixels and the millimetre
map (see `fit_mapping` for which kind of mapping is used) and reports how far each
clicked point is from where the map says it should be (the residual, in mm).

Note: the mapping is exact on the flat die face. The product sticks out toward the
camera, so its *width* (close to the face) is measured best; its *length* is the
length as seen on the die face.
"""
import cv2
import numpy as np

# Measured on the cold die (lab table, October 2026)
R_HOLE_MM = 31.0        # plate centre -> die hole centre
R_BOLT_MM = 45.0        # plate centre -> bolt centre
HOLE_A_TO_B_MM = 55.0
HOLE_A_TO_NEAR_BOLT_MM = 16.0
BOLT_TO_BOLT_MM = 65.0
DIE_HOLE_DIAMETER_MM = 4.28
BOLT_HEAD_DIAMETER_MM = 14.0

POINT_NAMES = ["hole_A", "bolt_near", "bolt_far", "hole_B"]
POINT_LABELS = {"hole_A": "Die hole A (product exit we track)",
                "bolt_near": "Bolt just below hole A (16 mm away)",
                "bolt_far": "Other bolt (65 mm from the near bolt)",
                "hole_B": "Die hole B (optional, often dark)"}


def _angle_deg(chord, r1, r2):
    """Angle at the plate centre between two points at radii r1, r2 that are `chord` apart."""
    return np.degrees(np.arccos((r1 ** 2 + r2 ** 2 - chord ** 2) / (2 * r1 * r2)))


def die_map_mm():
    """Positions (mm) of the reference points, plate centre at (0, 0), y pointing down as in images.

    The sides (which way round the points go) were chosen to match the video: with all
    four points the fit to the blank-die video leaves less than 2 px (about 0.5 mm) error.
    """
    def at(r, deg):
        return np.array([r * np.cos(np.radians(deg)), r * np.sin(np.radians(deg))])
    a_ab = _angle_deg(HOLE_A_TO_B_MM, R_HOLE_MM, R_HOLE_MM)          # 125.0 deg
    a_al = _angle_deg(HOLE_A_TO_NEAR_BOLT_MM, R_HOLE_MM, R_BOLT_MM)   # 11.9 deg
    a_lu = _angle_deg(BOLT_TO_BOLT_MM, R_BOLT_MM, R_BOLT_MM)          # 92.5 deg
    return {"plate_centre": np.zeros(2),
            "hole_A": at(R_HOLE_MM, 0.0),
            "hole_B": at(R_HOLE_MM, -a_ab),
            "bolt_near": at(R_BOLT_MM, a_al),
            "bolt_far": at(R_BOLT_MM, a_al - a_lu)}


def _similarity(P, Q):
    """Best scale + rotation + shift (no stretch, no mirror) taking P onto Q (least squares)."""
    mp, mq = P.mean(0), Q.mean(0)
    p, q = P - mp, Q - mq
    a = (p * q).sum()                                   # sum of dot products
    b = (p[:, 0] * q[:, 1] - p[:, 1] * q[:, 0]).sum()   # sum of cross products
    s = np.hypot(a, b) / (p ** 2).sum()
    th = np.arctan2(b, a)
    R = s * np.array([[np.cos(th), -np.sin(th)], [np.sin(th), np.cos(th)]])
    return np.hstack([R, (mq - R @ mp)[:, None]])


def fit_mapping(points_px, use=("bolt_near", "bolt_far")):
    """Fit pixels -> mm from clicked points.

    points_px : dict name -> (x, y) in full-frame pixels (any of POINT_NAMES)
    use       : which points define the mapping. The others are only *checked*.

    Why only the bolts by default: in a video with product, the die holes are always
    covered by the rope, and the rope sticks out toward the (tilted) camera, so the
    hole we see is shifted by about 2 mm from the real hole on the plate. The bolt
    heads are the only true plate points we can see, so they set the scale.
    Use all four points only in a video where the holes themselves are visible
    (for example a few seconds recorded before the product starts).

    Kind of mapping:
      * 2-3 points: "similarity" = one scale + rotation + shift. It ignores the small
        stretch caused by the camera angle (about +/-3 % on our blank-die video).
        (Hole A and the bolts lie almost on one line, so they cannot measure that stretch.)
      * 4 points: "affine" = scale + rotation + stretch.

    Returns dict with the 2x3 matrix `px_to_mm`, its inverse `mm_to_px`, the distance
    (mm) between every clicked point and where the map puts it, and px per mm.
    """
    ref = die_map_mm()
    names = [n for n in POINT_NAMES if n in use and points_px.get(n) is not None]
    if len(names) < 2:
        raise ValueError(f"Need at least two of {list(use)} to convert pixels to mm.")
    P = np.array([points_px[n] for n in names], float)
    Q = np.array([ref[n] for n in names], float)
    if len(names) >= 4:
        X = np.hstack([P, np.ones((len(P), 1))])
        M, *_ = np.linalg.lstsq(X, Q, rcond=None)
        px_to_mm, kind = M.T, "affine (scale + rotation + stretch)"
    else:
        px_to_mm, kind = _similarity(P, Q), "similarity (one scale + rotation)"
    mm_to_px = cv2.invertAffineTransform(px_to_mm)
    clicked = [n for n in POINT_NAMES if points_px.get(n) is not None]
    off = {n: float(np.linalg.norm(apply(px_to_mm, points_px[n]) - ref[n])) for n in clicked}
    lin = px_to_mm[:, :2]
    sv = np.linalg.svd(lin, compute_uv=False)     # mm per px along the two main directions
    return dict(points_used=names, kind=kind, px_to_mm=px_to_mm, mm_to_px=mm_to_px,
                offset_mm=off,
                px_per_mm=float(1 / np.sqrt(abs(np.linalg.det(lin)))),
                px_per_mm_range=(float(1 / sv.max()), float(1 / sv.min())))


def apply(matrix, xy):
    """Apply a 2x3 mapping to points (..., 2)."""
    xy = np.asarray(xy, float)
    return xy @ matrix[:, :2].T + matrix[:, 2]


def choose_reference(points_px):
    """Which points set the mm scale for this video, from what was clicked.

    1. both bolts visible -> the bolts (65 mm apart; true plate points);
    2. otherwise holes A and B (55 mm apart). Both are found from their rope ends, which stick
       out toward the camera by the same amount, so the shift cancels in their distance;
    3. otherwise None (pixels only; the scale must then come from the lab data).
    """
    have = {k for k, v in points_px.items() if v is not None}
    if {"bolt_near", "bolt_far"} <= have:
        return ("bolt_near", "bolt_far"), "bolts"
    if {"hole_A", "hole_B"} <= have:
        return ("hole_A", "hole_B"), "holes A+B"
    return None, "none"

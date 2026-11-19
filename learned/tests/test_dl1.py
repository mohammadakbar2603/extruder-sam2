"""Tests for learned_pathway.dl1 (DL-1): crops, clip features, z-score -> PCA -> ridge.

The leakage tests (T7, T8) must fail if any part of the model is fitted on clips that the
outer round holds out.
"""
import pytest

np = pytest.importorskip("numpy")
cv2 = pytest.importorskip("cv2")

from learned_pathway import pathway2 as p2  # noqa: E402
from learned_pathway import dl1  # noqa: E402

CLIPS = [f"{s}_{i}" for s in ("901", "902", "903", "904") for i in range(1, 6)]
THETA = np.deg2rad(30.0)
SCALE_MM_PER_PX = 1.0 / 3.4


def rect_mask(rows=(129, 230), cols=(165, 195), shape=(360, 360)) -> np.ndarray:
    """A filled rectangle mask, inclusive bounds."""
    m = np.zeros(shape, bool)
    m[rows[0]:rows[1] + 1, cols[0]:cols[1] + 1] = True
    return m


def sim_px_to_mm(scale: float = SCALE_MM_PER_PX, deg: float = 30.0) -> np.ndarray:
    """A 2 x 3 similarity (full-frame px -> mm): scale `scale` mm per px, rotated by `deg` degrees."""
    th = np.deg2rad(deg)
    rot = np.array([[np.cos(th), -np.sin(th)], [np.sin(th), np.cos(th)]])
    return np.hstack([scale * rot, np.zeros((2, 1))])


def make_F_y(seed: int = 0, d: int = 12) -> tuple[dict[str, np.ndarray], dict[str, float]]:
    rng = np.random.default_rng(seed)
    F = {c: rng.normal(size=d) for c in CLIPS}
    y = {c: float(rng.normal()) for c in CLIPS}
    return F, y


def test_t1_crop_scale_orientation_and_centring():
    """T1: the crop is drawn at 4 px per mm, upright (rotation removed) and centred on the centroid."""
    mask = rect_mask()  # 102 px tall, 31 px wide
    A = dl1.crop_affine(mask, sim_px_to_mm())
    outline, _, _ = dl1.make_crops(mask, A)
    assert outline.shape == (224, 224)
    ys, xs = np.nonzero(outline > 0)
    height, width = ys.max() - ys.min() + 1, xs.max() - xs.min() + 1
    assert height == pytest.approx(102 * 4.0 / 3.4, abs=2)  # ~120 px
    assert width == pytest.approx(31 * 4.0 / 3.4, abs=2)    # ~36.5 px
    assert height > 3 * width  # still upright: no rotation was applied
    assert xs.mean() == pytest.approx(111.5, abs=0.5)
    assert ys.mean() == pytest.approx(111.5, abs=0.5)


def test_t2_crop_values():
    """T2: the outline holds only 0/255; the colour crop is the warped frame inside, 0 outside."""
    mask = rect_mask()
    A = dl1.crop_affine(mask, sim_px_to_mm())
    outline, colour_none, _ = dl1.make_crops(mask, A)
    assert colour_none is None
    assert set(np.unique(outline)) <= {0, 255}
    frame = np.random.default_rng(1).integers(0, 256, (360, 360, 3), dtype=np.uint8)
    outline2, colour, _ = dl1.make_crops(mask, A, frame)
    assert np.array_equal(outline2, outline)
    warped = cv2.warpAffine(frame, A, (224, 224), flags=cv2.INTER_LINEAR,
                            borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    inside = outline > 0
    assert (colour[~inside] == 0).all()
    assert (colour[inside] == warped[inside]).all()


def test_t3_border_flag():
    """T3: a piece longer than the 56 mm crop touches the border; T1's does not."""
    long_mask = rect_mask(rows=(70, 289), cols=(170, 190))  # 220 px ~ 64.7 mm at 3.4 px/mm
    A = dl1.crop_affine(long_mask, sim_px_to_mm())
    _, _, touches = dl1.make_crops(long_mask, A)
    assert touches is True
    A1 = dl1.crop_affine(rect_mask(), sim_px_to_mm())
    _, _, touches1 = dl1.make_crops(rect_mask(), A1)
    assert touches1 is False


def test_t4_empty_mask_raises():
    with pytest.raises(ValueError):
        dl1.crop_affine(np.zeros((360, 360), bool), sim_px_to_mm())


def test_t16_loose_specks_are_dropped_before_cropping():
    """T16 (added at review): like extruder's measure_grid_mask, only the largest connected part
    (8-connected) is kept, so a far speck can neither shift the centre nor reach the border."""
    piece = rect_mask()
    with_speck = piece.copy()
    with_speck[10:13, 10:13] = True  # a 3 x 3 speck about 47 mm from the piece
    with_speck[128, 196] = True      # touches the corner (129, 195) only diagonally: part of it
    kept = dl1.largest_component(with_speck)
    expected = piece.copy()
    expected[128, 196] = True
    assert np.array_equal(kept, expected)
    A = dl1.crop_affine(kept, sim_px_to_mm())
    _, _, touches = dl1.make_crops(kept, A)
    assert touches is False
    assert not dl1.largest_component(np.zeros((5, 5), bool)).any()  # empty stays empty


def test_t5_clip_features():
    """T5: per-clip column means then ddof=1 SDs, clips in sorted order; 1-piece clips are refused."""
    feats = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]])
    clips = ["902_1", "902_1", "901_1", "901_1"]
    ids, X = dl1.clip_features(feats, clips)
    assert ids == ["901_1", "902_1"]
    assert X.shape == (2, 4)
    assert X[0] == pytest.approx([6.0, 7.0, np.std([5.0, 7.0], ddof=1), np.std([6.0, 8.0], ddof=1)])
    assert X[1] == pytest.approx([2.0, 3.0, np.std([1.0, 3.0], ddof=1), np.std([2.0, 4.0], ddof=1)])
    with pytest.raises(ValueError):
        dl1.clip_features(feats[:1], ["901_1"])


def test_t6_zpca_ridge_recovers_a_linear_rule():
    """T6: with k = d and a tiny alpha, z-score -> PCA -> ridge is ordinary least squares."""
    rng = np.random.default_rng(0)
    X = rng.normal(size=(40, 10))
    w = rng.normal(size=10)
    y = 3.0 + X @ w
    model = dl1.ZPCARidge(k=10, alpha=1e-9).fit(X, y)
    assert model.predict(X) == pytest.approx(y, abs=1e-6)
    X2 = rng.normal(size=(20, 10))
    assert model.predict(X2) == pytest.approx(3.0 + X2 @ w, abs=1e-6)


def test_t7_no_label_leakage():
    """T7: absurd labels on the held-out setting's clips cannot change that setting's predictions."""
    F, y = make_F_y()
    outer = p2.setting_folds(CLIPS)
    for scheme in ("setting", "clip"):
        base, _ = dl1.cross_predict_dl1(F, y, outer, scheme)
        poisoned = dict(y)
        for c in CLIPS:
            if c.startswith("903"):
                poisoned[c] = 1e6
        again, _ = dl1.cross_predict_dl1(F, poisoned, outer, scheme)
        assert all(again[c] == pytest.approx(base[c]) for c in CLIPS if c.startswith("903"))


def test_t8_no_feature_leakage():
    """T8: changed features of one held-out clip cannot change the other held-out clips' predictions."""
    F, y = make_F_y()
    outer = p2.setting_folds(CLIPS)
    for scheme in ("setting", "clip"):
        base, _ = dl1.cross_predict_dl1(F, y, outer, scheme)
        changed = {c: v.copy() for c, v in F.items()}
        changed["903_1"] = np.random.default_rng(5).normal(size=len(F["903_1"])) * 100.0
        again, _ = dl1.cross_predict_dl1(changed, y, outer, scheme)
        for c in ("903_2", "903_3", "903_4", "903_5"):
            assert again[c] == pytest.approx(base[c])


def test_t9_feasibility():
    """T9: k must fit inside the training folds; ZPCARidge refuses k >= n."""
    F, y = make_F_y(d=20)
    X15 = np.vstack([F[c] for c in CLIPS[:15]])
    y15 = np.array([y[c] for c in CLIPS[:15]])
    with pytest.raises(ValueError):
        dl1.ZPCARidge(15, 1.0).fit(X15, y15)
    _, _, table = dl1.select_k_alpha(F, y, CLIPS[:15], "setting")
    ks = {c["k"] for c in table}
    assert 16 not in ks  # the inner training folds have 10 clips, so k = 16 is impossible
    assert 8 in ks


def test_t10_tie_break():
    """T10: a constant target ties every candidate at MAE 0, so the smallest k and largest alpha win."""
    F, y = make_F_y()
    const_y = {c: 7.0 for c in CLIPS}
    k, alpha, table = dl1.select_k_alpha(F, const_y, CLIPS, "setting")
    assert k == 4 and alpha == 1000.0
    assert all(c["mae"] == pytest.approx(0.0) for c in table)


def test_t11_inner_folds():
    clips15 = CLIPS[:15]
    setting = dl1.inner_folds(clips15, "setting")
    assert len(setting) == 3 and all(len(f.test) == 5 for f in setting)
    clip = dl1.inner_folds(clips15, "clip")
    assert len(clip) == 15
    with pytest.raises(ValueError):
        dl1.inner_folds(clips15, "round-robin")


def test_t12_determinism_and_coverage():
    F, y = make_F_y()
    outer = p2.setting_folds(CLIPS)
    pred1, rec1 = dl1.cross_predict_dl1(F, y, outer, "setting")
    pred2, rec2 = dl1.cross_predict_dl1(F, y, outer, "setting")
    assert pred1 == pred2 and rec1 == rec2
    assert sorted(pred1) == sorted(CLIPS)  # every clip predicted exactly once
    assert len(rec1) == len(outer)
    assert {r["held_out"] for r in rec1} == {"901", "902", "903", "904"}
    twice = [p2.Round("a", ["901_1"], CLIPS[1:]), p2.Round("b", ["901_1"], CLIPS[:-1])]
    with pytest.raises(ValueError):
        dl1.cross_predict_dl1(F, y, twice, "setting")

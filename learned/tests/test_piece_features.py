"""Tests for learned_pathway.piece_features: crops at 4 px/mm, 8 views, cached features (SYNTHETIC)."""
import json
import os
from pathlib import Path

import pytest

np = pytest.importorskip("numpy")
pd = pytest.importorskip("pandas")
cv2 = pytest.importorskip("cv2")
torch = pytest.importorskip("torch")
Image = pytest.importorskip("PIL.Image")

from learned_pathway import piece_features as pf  # noqa: E402


class CountingEncoder:
    """A stand-in for DINOv2: (B, 3, H, W) -> (B, 768), the channel means repeated; counts calls."""

    def __init__(self):
        self.calls = 0

    def __call__(self, x):
        self.calls += 1
        return x.mean((2, 3)).repeat(1, 256)


def write_synthetic(root: Path, n: int = 3) -> pd.DataFrame:
    """One SYNTHETIC clip with n rectangle masks of different sizes, and its clicks (3.4 px/mm)."""
    clip = "SYN-a_1"
    d = root / clip
    (d / "masks").mkdir(parents=True)
    pts = {"hole_A": [150.0, 150.0], "bolt_near": [100.0, 200.0], "bolt_far": [100.0 + 65 * 3.4, 200.0]}
    (d / "clicks.json").write_text(json.dumps({"clicked": pts, "refined": pts}))
    rows = []
    for i in range(n):
        m = np.zeros((360, 360), np.uint8)
        m[150:170 + 10 * i, 170:180] = 255
        path = d / "masks" / f"cut{i:04d}_red_f{10 * i:05d}.png"
        Image.fromarray(m).save(path)
        rows.append(dict(clip=clip, setting="SYN-a", cut=i, key="red", frame=10 * i, used=True, repeat=False,
                         mask_path=str(path), origin_x=600, origin_y=300))
    return pd.DataFrame(rows)


def test_dihedral_views():
    img = np.arange(16).reshape(4, 4)  # crops are square; an asymmetric one has 8 distinct views
    v = pf.dihedral(img)
    assert v.shape == (8, 4, 4)
    assert np.array_equal(v[0], img)
    assert np.array_equal(v[4], np.fliplr(img))
    assert len({x.tobytes() for x in v}) == 8


def test_dihedral_views_colour():
    img = np.random.default_rng(0).integers(0, 255, (5, 5, 3), dtype=np.uint8)
    v = pf.dihedral(img)
    assert v.shape == (8, 5, 5, 3)
    assert np.array_equal(v[1], np.rot90(img, 1))


def test_piece_crop_outline_scale():
    m = np.zeros((360, 360), bool)
    m[100:140, 170:180] = True  # 40 x 10 px
    A = np.array([[0.25, 0, 0], [0, 0.25, 0]])  # 4 px per mm: the crop keeps the size
    out = pf.piece_crop(m, A, None)
    assert out.shape == (224, 224) and out.dtype == np.uint8
    assert (out > 0).sum() == pytest.approx(400, rel=0.03)


def test_piece_crop_colour():
    m = np.zeros((360, 360), bool)
    m[100:140, 170:180] = True
    frame = np.full((360, 360, 3), 200, np.uint8)
    out = pf.piece_crop(m, np.array([[0.25, 0, 0], [0, 0.25, 0]]), frame)
    assert out.shape == (224, 224, 3)
    assert out[112, 112].tolist() == [200, 200, 200] and out[0, 0].tolist() == [0, 0, 0]


def test_build_features_shape_and_order(tmp_path):
    pieces = write_synthetic(tmp_path)
    index, F = pf.build_features(pieces, tmp_path, "outline", tmp_path / "cache", encoder=CountingEncoder())
    assert F.shape == (3, 8, 768) and F.dtype == np.float32
    assert index["cut"].tolist() == [0, 1, 2]
    assert "touches_border" in index.columns
    # bigger masks -> more white -> larger mean (the fake encoder's output), in input order
    assert F[0, 0, 0] < F[1, 0, 0] < F[2, 0, 0]


def test_build_features_uses_cache_and_rebuilds(tmp_path):
    pieces = write_synthetic(tmp_path)
    enc = CountingEncoder()
    _, F1 = pf.build_features(pieces, tmp_path, "outline", tmp_path / "cache", encoder=enc)
    n = enc.calls
    assert n > 0
    _, F2 = pf.build_features(pieces, tmp_path, "outline", tmp_path / "cache", encoder=enc)
    assert enc.calls == n and np.array_equal(F1, F2)
    p = Path(pieces["mask_path"].iloc[0])
    st = p.stat()
    os.utime(p, ns=(st.st_atime_ns, st.st_mtime_ns + 10**9))
    pf.build_features(pieces, tmp_path, "outline", tmp_path / "cache", encoder=enc)
    assert enc.calls > n


def test_colour_frames_reads_right_frame(tmp_path):
    path = tmp_path / "SYN.mp4"
    w = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 10, (64, 48))
    if not w.isOpened():
        pytest.skip("no mp4 writer in this OpenCV build")
    for i in range(5):
        w.write(np.full((48, 64, 3), 40 * i, np.uint8))
    w.release()
    got = pf.colour_frames(path, [1, 3], origin=(8, 4), size=32)
    assert sorted(got) == [1, 3]
    assert got[1].shape == (32, 32, 3)
    assert got[1].mean() == pytest.approx(40, abs=6) and got[3].mean() == pytest.approx(120, abs=6)


def test_bags_follow_piece_set_rows():
    F = np.arange(5 * 8 * 768, dtype=np.float32).reshape(5, 8, 768)
    pset = pd.DataFrame({"clip": ["SYN-a_1", "SYN-b_1", "SYN-a_1"], "row": [4, 0, 2]})
    B = pf.bags(pset, F)
    assert list(B) == ["SYN-a_1", "SYN-b_1"]
    assert np.array_equal(B["SYN-a_1"], F[[4, 2]]) and np.array_equal(B["SYN-b_1"], F[[0]])


def test_crops_of_outline(tmp_path):
    pieces = write_synthetic(tmp_path)
    crops = pf.crops_of(pieces.iloc[[2, 0]], tmp_path, "outline")
    assert len(crops) == 2 and crops[0].shape == (224, 224)
    assert (crops[0] > 0).sum() > (crops[1] > 0).sum()  # piece 2 is the longest

"""Tests for learned_pathway.nb02: reading notebook 02's results for notebook 03 (SYNTHETIC folders)."""
import json
from pathlib import Path

import pytest

np = pytest.importorskip("numpy")
pd = pytest.importorskip("pandas")

from learned_pathway import nb02  # noqa: E402

OTHER_COMPUTER = "/cluster/home/someone/project/results"
NAN = float("nan")


def write_clip(root: Path, clip: str, red_frames: list[float], used: list[bool], origin=(600, 300),
               bolts=((100.0, 200.0), (100.0 + 65 * 3.4, 200.0))) -> list[dict]:
    """One SYNTHETIC clip folder: cuts.csv, clicks.json, crop_frames/meta.json, masks/ (empty files).
    Returns the clip's experiment1/pieces.csv rows (RED and BLUE for every cut with a RED frame)."""
    d = root / clip
    (d / "masks").mkdir(parents=True)
    (d / "crop_frames").mkdir()
    (d / "crop_frames" / "meta.json").write_text(json.dumps({"video": f"{clip}.MP4", "origin": list(origin),
                                                              "crop_size": 360}))
    pts = {"hole_A": [150.0, 150.0], "bolt_near": list(bolts[0]), "bolt_far": list(bolts[1])}
    (d / "clicks.json").write_text(json.dumps({"video": f"{clip}.MP4", "clicked": pts, "refined": pts}))
    cuts = pd.DataFrame({"cut": range(len(red_frames)), "event_frame": [10 * i for i in range(len(red_frames))],
                         "red_frame": red_frames,
                         "blue_frame": [r + 1 if r == r else NAN for r in red_frames],
                         "keep": [r == r for r in red_frames]})
    cuts.to_csv(d / "cuts.csv", index=False)
    rows = []
    for cut, (r, u) in enumerate(zip(red_frames, used)):
        if r != r:
            continue
        for key, f in (("red", int(r)), ("blue", int(r) + 1)):
            name = f"cut{cut:04d}_{key}_f{f:05d}.png"
            (d / "masks" / name).write_bytes(b"")
            rows.append(dict(video=clip, clip=clip, setting=clip.split("_")[0], cut=cut, key=key, frame=f,
                             used=u, mask_problem="" if u else "ragged_outline",
                             length_mm=20.0 + cut, width_mm=9.0 + cut,
                             mask_file=f"{OTHER_COMPUTER}/{clip}/masks/{name}"))
    return rows


def write_results(root: Path, clips: dict[str, tuple[list[float], list[bool]]]) -> Path:
    rows = []
    for clip, (reds, used) in clips.items():
        rows += write_clip(root, clip, reds, used)
    (root / "experiment1").mkdir()
    pd.DataFrame(rows).to_csv(root / "experiment1" / "pieces.csv", index=False)
    return root


def test_repeat_cuts():
    cuts = pd.DataFrame({"cut": [0, 1, 2, 3, 4, 5], "red_frame": [13, 24, 24, 35, NAN, NAN]})
    assert nb02.repeat_cuts(cuts) == {2}


def test_repeat_cuts_uses_cut_order():
    cuts = pd.DataFrame({"cut": [2, 0, 1], "red_frame": [24, 13, 24]})
    assert nb02.repeat_cuts(cuts) == {2}


def test_local_mask_path(tmp_path):
    p = nb02.local_mask_path(tmp_path, "/cluster/home/someone/project/results/SYN-a_1/masks/cut0001_red_f00024.png")
    assert p == tmp_path / "SYN-a_1" / "masks" / "cut0001_red_f00024.png"


def test_load_pieces_adds_columns(tmp_path):
    root = write_results(tmp_path, {"SYN-a_1": ([13, 24, 24, 35], [True] * 4),
                                    "SYN-b_1": ([5, 16, 27], [True, False, True])})
    p = nb02.load_pieces(root)
    assert {"mask_path", "repeat", "origin_x", "origin_y"} <= set(p.columns)
    assert all(Path(m).exists() for m in p["mask_path"])
    rep = p[p["repeat"]]
    assert set(zip(rep["clip"], rep["cut"])) == {("SYN-a_1", 2)}
    assert len(rep) == 2  # RED and BLUE of the repeated cut
    assert (p["origin_x"] == 600).all() and (p["origin_y"] == 300).all()


def test_piece_set_filters_used_and_repeats(tmp_path):
    root = write_results(tmp_path, {"SYN-a_1": ([13, 24, 24, 35], [True] * 4),
                                    "SYN-b_1": ([5, 16, 27], [True, False, True])})
    s = nb02.piece_set(nb02.load_pieces(root), "red", ["SYN-a_1", "SYN-b_1"])
    assert sorted(zip(s["clip"], s["cut"])) == [("SYN-a_1", 0), ("SYN-a_1", 1), ("SYN-a_1", 3),
                                                ("SYN-b_1", 0), ("SYN-b_1", 2)]
    assert set(s["key"]) == {"red"}


def test_piece_set_refuses_thin_clips(tmp_path):
    root = write_results(tmp_path, {"SYN-a_1": ([13, 24, 35], [True] * 3),
                                    "SYN-b_1": ([5, 16], [True, False])})
    with pytest.raises(ValueError, match=r"SYN-b_1.*red|red.*SYN-b_1"):
        nb02.piece_set(nb02.load_pieces(root), "red", ["SYN-a_1", "SYN-b_1"])


def test_clip_ruler_from_bolts(tmp_path):
    root = write_results(tmp_path, {"SYN-a_1": ([13, 24], [True, True])})
    A = nb02.clip_ruler(root, "SYN-a_1")
    assert A.shape == (2, 3)
    scale = np.sqrt(abs(np.linalg.det(A[:, :2])))  # mm per px
    assert scale == pytest.approx(1 / 3.4, rel=1e-6)


def test_matched_clips():
    lab = pd.DataFrame({"clip": ["SYN-a_1", "SYN-a_2", "SYN-b_1"]})
    pieces = pd.DataFrame({"clip": ["SYN-a_1", "SYN-b_1", "SYN-c_1", "SYN-c_1"]})
    assert nb02.matched_clips(lab, pieces) == (["SYN-a_1", "SYN-b_1"], ["SYN-a_2", "SYN-c_1"])


def test_pathway1_means():
    pset = pd.DataFrame({"clip": ["SYN-a_1", "SYN-a_1"], "width_mm": [9.0, 11.0], "length_mm": [20.0, 30.0]})
    m = nb02.pathway1_means(pset)
    assert m == {"diameter": {"SYN-a_1": 10.0}, "length": {"SYN-a_1": 25.0}}


def test_lab_table_reads_sheet_with_more_pieces(tmp_path):
    """A sheet with 10 pieces per clip gives n = 10 and the right means."""
    rows = [{"Sl. No.": "SYN-a_1", "Piece no.": i, "Length of the Product (mm)": 20.0 + i,
             "Diameter of the Product (mm)": 9.0} for i in range(1, 11)]
    path = tmp_path / "labels.csv"
    pd.DataFrame(rows).to_csv(path, index=False)
    lab = nb02.lab_table(path)
    r = lab.set_index("clip").loc["SYN-a_1"]
    assert r["lab_length_n"] == 10 and r["lab_length_mean"] == pytest.approx(25.5)
    assert r["lab_diameter_mean"] == pytest.approx(9.0)


def test_piece_set_keeps_row_of_pieces(tmp_path):
    root = write_results(tmp_path, {"SYN-a_1": ([13, 24, 24, 35], [True] * 4),
                                    "SYN-b_1": ([5, 16, 27], [True, False, True])})
    pieces = nb02.load_pieces(root)
    s = nb02.piece_set(pieces, "red", ["SYN-a_1", "SYN-b_1"])
    back = pieces.loc[s["row"].to_numpy()]
    assert back["clip"].tolist() == s["clip"].tolist() and back["cut"].tolist() == s["cut"].tolist()
    assert (back["key"] == "red").all()


def test_piece_set_row_is_position_for_a_filtered_table(tmp_path):
    # features are stored by position (build_features resets the index), so `row` must be a position
    root = write_results(tmp_path, {"SYN-a_1": ([13, 24, 24, 35], [True] * 4),
                                    "SYN-b_1": ([5, 16, 27], [True, False, True])})
    pieces = nb02.load_pieces(root)
    sub = pieces[pieces["clip"].eq("SYN-b_1")]  # index labels do not start at 0
    s = nb02.piece_set(sub, "red", ["SYN-b_1"])
    back = sub.iloc[s["row"].to_numpy()]
    assert back["cut"].tolist() == s["cut"].tolist() and (back["key"] == "red").all()

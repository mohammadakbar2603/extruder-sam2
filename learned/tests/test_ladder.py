"""Tests for learned_pathway.ladder: the comparison table, metrics, paired wins and figures (SYNTHETIC)."""
import pytest

np = pytest.importorskip("numpy")
pd = pytest.importorskip("pandas")

from learned_pathway import ladder  # noqa: E402

CLIPS = [f"SYN-{s}_{i}" for s in "abcd" for i in range(1, 6)]
LOSO = "leave one setting out"


def lab_frame(values=None, se=0.1) -> pd.DataFrame:
    values = values or {c: 9.0 for c in CLIPS}
    return pd.DataFrame({"clip": list(values), "lab_diameter_mean": list(values.values()),
                         "lab_diameter_sd": 0.3, "lab_diameter_se": se,
                         "lab_length_mean": 30.0, "lab_length_sd": 2.0, "lab_length_se": 1.0})


def test_rows_error_sign():
    r = ladder.rows(LOSO, "DL-2", "outline", "red", "diameter", {"SYN-a_1": 10.0}, lab_frame())
    assert len(r) == 1
    row = r.iloc[0]
    assert row["error"] == pytest.approx(1.0) and row["setting"] == "SYN-a"
    assert row["variant"] == "DL-2" and row["version"] == "outline" and np.isnan(row["video_sd"])


def test_sei_rows():
    lab = lab_frame({"SYN-a_1": 4.28})
    d = ladder.rows(LOSO, "B0", "none", "red", "diameter", {"SYN-a_1": 8.56}, lab)
    s = ladder.sei_rows(d).iloc[0]
    assert s["target"] == "SEI"
    assert s["video"] == pytest.approx(4.0, abs=1e-12)
    assert s["lab"] == pytest.approx(1.0, abs=1e-12)
    assert s["lab_se"] == pytest.approx(2 * 4.28 * 0.1 / 4.28 ** 2)
    assert s["error"] == pytest.approx(3.0)


def comp_with(method_err: float, yard_err: float) -> pd.DataFrame:
    lab = lab_frame()
    a = ladder.rows(LOSO, "DL-2", "outline", "red", "diameter", {c: 9.0 + method_err for c in CLIPS}, lab)
    b = ladder.rows(LOSO, "B0", "none", "red", "diameter", {c: 9.0 + yard_err for c in CLIPS}, lab)
    return pd.concat([a, b], ignore_index=True)


def test_paired_table_win_flag():
    p = ladder.paired_table(comp_with(0.1, 1.0))
    row = p[(p["method"] == "DL-2") & (p["yardstick"] == "B0")].iloc[0]
    assert bool(row["wins"]) and row["mae_difference"] == pytest.approx(-0.9)
    p2 = ladder.paired_table(comp_with(1.0, 0.1))
    assert not bool(p2[(p2["method"] == "DL-2") & (p2["yardstick"] == "B0")].iloc[0]["wins"])


def test_paired_table_skips_self():
    p = ladder.paired_table(comp_with(0.1, 1.0))
    assert not ((p["method"] == "B0") & (p["yardstick"] == "B0")).any()


def test_metric_table_per_setting():
    pred = {c: 9.0 + (0.5 if c.startswith("SYN-a") else 0.1) for c in CLIPS}
    comp = ladder.rows(LOSO, "B0", "none", "red", "diameter", pred, lab_frame())
    m = ladder.metric_table(comp).iloc[0]
    assert m["MAE_SYN-a"] == pytest.approx(0.5) and m["MAE_SYN-b"] == pytest.approx(0.1)
    assert m["MAE"] == pytest.approx((0.5 * 5 + 0.1 * 15) / 20)
    assert m["scheme"] == LOSO and m["version"] == "none"


def test_figures_write_files(tmp_path):
    lab = lab_frame()
    comp = pd.concat([ladder.rows(LOSO, m, v, "red", t, {c: 9.2 for c in CLIPS}, lab)
                      for m, v in (("B0", "none"), ("DL-2", "outline")) for t in ("diameter", "length")])
    comp = pd.concat([comp, ladder.sei_rows(comp[comp["target"] == "diameter"])], ignore_index=True)
    met = ladder.metric_table(comp)
    paths = [ladder.plot_ladder(met, tmp_path / "ladder.png"),
             ladder.plot_pred_vs_lab(comp, "DL-2", "red", "outline", tmp_path / "pvl.png"),
             ladder.gallery([np.zeros((224, 224), np.uint8)] * 3, ["a", "b", "c"], tmp_path / "g.png", "t"),
             ladder.plot_training_curve([1.0, 0.5, 0.2], tmp_path / "curve.png")]
    for p in paths:
        assert p.exists() and p.stat().st_size > 1000


def test_setting_permutations_are_the_24_orders():
    maps = ladder.setting_permutations(CLIPS)
    assert len(maps) == 24
    assert len({tuple(sorted(m.items())) for m in maps}) == 24
    assert maps[0] == {c: c for c in CLIPS}  # identity first: the notebook reads the true MAE from row 0
    for m in maps:
        assert sorted(m) == sorted(CLIPS) and sorted(m.values()) == sorted(CLIPS)
        # clip i of a setting takes the label of clip i of one other setting; settings move as blocks
        for s in "abcd":
            targets = {m[f"SYN-{s}_{i}"] for i in range(1, 6)}
            src = {t.rsplit("_", 1)[0] for t in targets}
            assert len(src) == 1
            assert all(m[f"SYN-{s}_{i}"].endswith(f"_{i}") for i in range(1, 6))


def test_permutation_p_counts_ties_and_identity():
    assert ladder.permutation_p(1.0, [0.5, 1.0, 2.0, 3.0]) == 0.5
    assert ladder.permutation_p(0.1, [0.1, 2.0, 3.0, 4.0]) == 0.25

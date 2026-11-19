"""Tests for learned_pathway.pathway2: B0 and DL-0 under leave one setting out."""
import math

import pytest

from learned_pathway import pathway2 as p2

CLIPS = [f"{s}_{i}" for s in ("901", "902", "903", "904") for i in range(1, 6)]


def test_setting_folds_hold_out_each_setting_whole_and_test_every_clip_once():
    folds = p2.setting_folds(CLIPS)
    assert [f.held_out for f in folds] == ["901", "902", "903", "904"]
    tested = [c for f in folds for c in f.test]
    assert sorted(tested) == sorted(CLIPS)
    for f in folds:
        assert all(c.startswith(f.held_out) for c in f.test)
        assert not any(c.startswith(f.held_out) for c in f.train)
        assert len(f.test) == 5 and len(f.train) == 15


def test_clip_folds_leave_out_one_clip_at_a_time():
    folds = p2.clip_folds(CLIPS)
    assert len(folds) == 20
    for f in folds:
        assert f.test == [f.held_out] and f.held_out not in f.train and len(f.train) == 19


def test_dl0_factor_is_mean_lab_over_mean_video():
    assert p2.dl0_factor(lab=[9.0, 9.6], video=[10.0, 12.0]) == pytest.approx(9.3 / 11.0)


def test_b0_and_dl0_predictions_ignore_the_held_out_labels():
    video = {c: 10.0 + 0.1 * k for k, c in enumerate(CLIPS)}
    lab = {c: 9.0 + 0.05 * k for k, c in enumerate(CLIPS)}
    folds = p2.setting_folds(CLIPS)
    for method in ("B0", "DL-0"):
        base = p2.cross_predict(video, lab, folds, method)
        changed = dict(lab)
        for c in folds[2].test:  # the 903 clips get absurd lab values
            changed[c] = 1000.0
        again = p2.cross_predict(video, changed, folds, method)
        assert all(again[c] == pytest.approx(base[c]) for c in folds[2].test)


def test_b0_predicts_the_mean_of_the_training_clips():
    video = {c: 10.0 for c in CLIPS}
    lab = {c: float(k) for k, c in enumerate(CLIPS)}
    pred = p2.cross_predict(video, lab, p2.setting_folds(CLIPS), "B0")
    train_901 = [lab[c] for c in CLIPS if not c.startswith("901")]
    assert pred["901_3"] == pytest.approx(sum(train_901) / len(train_901))


def test_dl0_predicts_k_times_the_clip_value():
    video = {c: 10.0 + k for k, c in enumerate(CLIPS)}
    lab = {c: 0.8 * v for c, v in video.items()}  # an exact scale error
    pred = p2.cross_predict(video, lab, p2.setting_folds(CLIPS), "DL-0")
    assert all(pred[c] == pytest.approx(lab[c]) for c in CLIPS)


def test_cross_predict_rejects_an_unknown_method_and_missing_clips():
    video = {c: 1.0 for c in CLIPS}
    with pytest.raises(ValueError):
        p2.cross_predict(video, dict(video), p2.setting_folds(CLIPS), "DL-9")
    with pytest.raises(KeyError):
        p2.cross_predict({c: 1.0 for c in CLIPS[1:]}, dict(video), p2.setting_folds(CLIPS), "DL-0")


def test_paired_difference_of_identical_errors_is_zero():
    e = [0.3, -0.1, 0.2, 0.4, -0.2]
    r = p2.paired_mae_difference(e, e)
    assert r["mae_difference"] == 0.0 and r["ci_low"] == 0.0 and r["ci_high"] == 0.0
    assert math.isnan(r["wilcoxon_p"])  # no non-zero differences to rank


def test_paired_difference_finds_a_clearly_better_method():
    a = [0.05, -0.04, 0.06, 0.03, -0.05, 0.04, 0.02, -0.06, 0.05, 0.03] * 2
    b = [0.60, -0.50, 0.70, 0.40, -0.55, 0.65, 0.45, -0.70, 0.50, 0.60] * 2
    r = p2.paired_mae_difference(a, b)
    assert r["mae_difference"] < 0 and r["ci_high"] < 0 and r["wilcoxon_p"] < 0.01
    assert p2.paired_mae_difference(a, b) == r  # fixed seed: same interval every time


# ---- the length rule (notebook 03): length = rope speed / cut rate ------------------

def test_cut_rate_hz():
    assert p2.cut_rate_hz(450) == 22.5
    assert p2.cut_rate_hz(1005) == 50.25


def test_length_rule_hand_example():
    rate = {"SYN-a_1": 10.0, "SYN-b_1": 20.0, "SYN-c_1": 40.0}
    length = {"SYN-a_1": 100.0, "SYN-b_1": 60.0, "SYN-c_1": 35.0}  # rope speeds 1000, 1200, 1400 mm/s
    pred = p2.length_rule(length, rate, p2.setting_folds(list(rate)))
    assert pred == pytest.approx({"SYN-a_1": 130.0, "SYN-b_1": 60.0, "SYN-c_1": 27.5}, abs=1e-12)


def test_length_rule_ignores_held_out_label():
    rate = {"SYN-a_1": 10.0, "SYN-b_1": 20.0, "SYN-c_1": 40.0}
    length = {"SYN-a_1": 100.0, "SYN-b_1": 60.0, "SYN-c_1": 35.0}
    folds = p2.setting_folds(list(rate))
    before = p2.length_rule(length, rate, folds)["SYN-a_1"]
    after = p2.length_rule({**length, "SYN-a_1": 999.0}, rate, folds)["SYN-a_1"]
    assert after == before

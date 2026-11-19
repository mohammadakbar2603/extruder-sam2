"""Tests for learned_pathway.dl2: the gated-attention network on frozen piece features (SYNTHETIC).

The leakage tests must fail if the held-out setting's labels or pieces reach any learned part:
the PCA, the target scaling, the network weights or the inner choice of weight decay and epochs.
"""
import pytest

np = pytest.importorskip("numpy")
torch = pytest.importorskip("torch")

from learned_pathway import dl2  # noqa: E402
from learned_pathway import pathway2 as p2  # noqa: E402

SETTINGS = ("SYN-a", "SYN-b", "SYN-c", "SYN-d")
FAST = dl2.Config(weight_decay=1e-2, epochs=300)


def planted(seed: int = 0, m: int = 80, d: int = 768):
    """SYNTHETIC bags: 4 settings x 5 clips x m pieces, features (m, 8, d).

    Clip size s ~ N(0, 1); label y = 10 + 2 s. Good pieces carry s on columns 0-4 (noise SD 0.3);
    bad pieces (5-40 per clip) carry +3 on columns 0-4 and a marker +3 on columns 5-9, so a plain
    mean over pieces is biased by the share of bad pieces, and attention can learn to skip them.
    Returns (bags, y, good) with good[clip] a boolean array over its pieces.
    """
    rng = np.random.default_rng(seed)
    bags, y, good = {}, {}, {}
    for s_name in SETTINGS:
        for i in range(1, 6):
            clip = f"{s_name}_{i}"
            s = rng.normal()
            n_bad = int(rng.integers(max(1, m // 16), m // 2 + 1))  # 5-40 of 80 pieces
            g = np.r_[np.ones(m - n_bad, bool), np.zeros(n_bad, bool)]
            x = rng.normal(size=(m, d)).astype(np.float32)
            x[g, :5] = s + rng.normal(scale=0.3, size=(g.sum(), 5))
            x[~g, :5] = 3.0 + rng.normal(scale=0.3, size=(n_bad, 5))
            x[~g, 5:10] += 3.0
            views = x[:, None, :] + rng.normal(scale=0.05, size=(m, 8, d)).astype(np.float32)
            bags[clip], y[clip], good[clip] = views.astype(np.float32), 10 + 2 * s, g
    return bags, y, good


def test_parameter_count():
    assert sum(p.numel() for p in dl2.GatedAttentionMIL().parameters()) == 1106


def test_attention_sums_to_one_and_ignores_order():
    torch.manual_seed(0)
    net = dl2.GatedAttentionMIL()
    bag = torch.randn(10, dl2.PCA_DIM)
    with torch.no_grad():
        y, w = net(bag)
        perm = torch.randperm(10)
        y2, w2 = net(bag[perm])
    assert y.shape == () and w.shape == (10,)
    assert float(w.sum()) == pytest.approx(1.0, abs=1e-6)
    assert float(y2) == pytest.approx(float(y), abs=1e-6)
    assert torch.allclose(w2, w[perm], atol=1e-6)


def test_padding_mask_matches_single_bag():
    torch.manual_seed(0)
    net = dl2.GatedAttentionMIL()
    a, b = torch.randn(7, dl2.PCA_DIM), torch.randn(4, dl2.PCA_DIM)
    x = torch.zeros(2, 7, dl2.PCA_DIM)
    x[0], x[1, :4] = a, b
    mask = torch.tensor([[True] * 7, [True] * 4 + [False] * 3])
    with torch.no_grad():
        yb, wb = net(x, mask)
        single = net(b)[0]
    assert float(yb[1]) == pytest.approx(float(single), abs=1e-6)
    assert float(wb[1, 4:].sum()) == 0.0


def test_pca_is_fitted_on_training_rows():
    X = np.random.default_rng(0).normal(size=(200, 768))
    pca = dl2.PiecePCA().fit(X)
    Z = pca.transform(X)
    assert Z.shape == (200, dl2.PCA_DIM)
    assert np.allclose(Z.mean(0), 0, atol=1e-5)  # float32 output
    assert pca.transform(np.zeros((3, 8, 768))).shape == (3, 8, dl2.PCA_DIM)


def test_recovers_planted_signal():
    bags, y, good = planted()
    clips = sorted(bags)
    folds = p2.setting_folds(clips)
    pred, records, attention = dl2.cross_predict_dl2(bags, y, folds, grid=(FAST,), seeds=(0, 1))
    b0 = p2.cross_predict({}, y, folds, "B0")
    mae = np.mean([abs(pred[c] - y[c]) for c in clips])
    mae_b0 = np.mean([abs(b0[c] - y[c]) for c in clips])
    assert mae < 0.5 * mae_b0
    w_good = np.mean([attention[c][good[c]].mean() for c in clips])
    w_bad = np.mean([attention[c][~good[c]].mean() for c in clips])
    assert w_good > w_bad
    assert [r["held_out"] for r in records] == list(SETTINGS)


def test_deterministic():
    bags, y, _ = planted(m=30)
    folds = p2.setting_folds(sorted(bags))
    cfg = dl2.Config(1e-2, 30)
    a = dl2.cross_predict_dl2(bags, y, folds, grid=(cfg,), seeds=(0,))
    b = dl2.cross_predict_dl2(bags, y, folds, grid=(cfg,), seeds=(0,))
    assert a[0] == b[0]
    assert [r["digest"] for r in a[1]] == [r["digest"] for r in b[1]]


def test_training_curve():
    bags, y, _ = planted(m=40)
    train = {c: v for c, v in bags.items() if not c.startswith("SYN-a")}
    losses = dl2.training_curve(train, y, dl2.Config(1e-2, 50), seed=0)
    assert len(losses) == 50
    assert losses[-1] < losses[0]


def held_out_records(select=None):
    """Two runs on SYNTHETIC data: as is, and with setting SYN-a's labels changed and its features
    tripled. Returns SYN-a's round record from each run."""
    bags, y, _ = planted(m=30)
    folds = p2.setting_folds(sorted(bags))
    grid = (dl2.Config(1e-3, 10), dl2.Config(1e-1, 20))
    r1 = dl2.cross_predict_dl2(bags, y, folds, grid=grid, seeds=(0,))[1][0]
    bags2 = {c: (3 * v if c.startswith("SYN-a") else v) for c, v in bags.items()}
    y2 = {c: (v + 50 if c.startswith("SYN-a") else v) for c, v in y.items()}
    r2 = dl2.cross_predict_dl2(bags2, y2, folds, grid=grid, seeds=(0,))[1][0]
    assert r1["held_out"] == r2["held_out"] == "SYN-a"
    return r1, r2


def assert_same_round(r1, r2):
    for k in ("weight_decay", "epochs", "inner_mae", "digest", "n_train"):
        assert r1[k] == r2[k], k


def test_no_leakage_from_held_out():
    assert_same_round(*held_out_records())


def test_leak_mutation_is_caught(monkeypatch):
    """Tuning on all clips (the held-out setting included) must be caught by the leakage test."""
    original = dl2.select_config

    def leaky(bags, y, clips, grid=dl2.GRID, seeds=dl2.SEEDS):
        return original(bags, y, sorted(bags), grid, seeds)

    monkeypatch.setattr(dl2, "select_config", leaky)
    with pytest.raises(AssertionError):
        assert_same_round(*held_out_records())


def test_pca_sees_training_pieces_only(monkeypatch):
    """The PCA inside fit_predict is fitted on the training bags' pieces (all views) and nothing else."""
    seen = []
    original = dl2.PiecePCA.fit

    def spy(self, X):
        seen.append(len(X))
        return original(self, X)

    monkeypatch.setattr(dl2.PiecePCA, "fit", spy)
    bags, y, _ = planted(m=30)
    train = {c: v for c, v in bags.items() if not c.startswith("SYN-a")}
    test = {c: v for c, v in bags.items() if c.startswith("SYN-a")}
    dl2.fit_predict(train, y, test, dl2.Config(1e-2, 5), seeds=(0,))
    assert seen == [sum(len(v) for v in train.values()) * 8]

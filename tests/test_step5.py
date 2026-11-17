import numpy as np

from possession import harness
from possession.external.degrade import CLEAN, PROFILES, degrade
from synth import STANDARD, make_match


def _long_match():
    tr, _ = make_match(STANDARD * 40)       # ~50k frames
    return tr, [(0, tr.n_frames - 1)]


def test_degrade_hits_the_documented_missing_share_for_every_profile():
    tr, blocks = _long_match()
    for name, prof in PROFILES.items():
        d = degrade(tr, blocks, prof, seed=1)
        share = np.isnan(d.ball_xy[:, 0]).mean()
        assert abs(share - 0.41) < 0.06, (name, share)


def test_degrade_gap_length_follows_the_profile():
    tr, blocks = _long_match()
    lens = {}
    for name in ("gap10", "gap200"):
        m = np.isnan(degrade(tr, blocks, PROFILES[name], seed=1).ball_xy[:, 0]).astype(int)
        edges = np.diff(np.concatenate(([0], m, [0])))
        lens[name] = (np.flatnonzero(edges == -1) - np.flatnonzero(edges == 1)).mean()
    assert lens["gap10"] < 25 < 100 < lens["gap200"]


def test_degrade_is_deterministic_and_does_not_touch_the_input():
    tr, blocks = _long_match()
    before = tr.ball_xy.copy()
    a = degrade(tr, blocks, PROFILES["gap25"], seed=3)
    b = degrade(tr, blocks, PROFILES["gap25"], seed=3)
    assert np.array_equal(a.ball_xy, b.ball_xy, equal_nan=True)
    assert np.array_equal(tr.ball_xy, before, equal_nan=True)


def test_clean_profile_changes_nothing():
    tr, blocks = _long_match()
    d = degrade(tr, blocks, CLEAN, seed=0)
    assert np.array_equal(d.ball_xy, tr.ball_xy, equal_nan=True)
    assert np.array_equal(d.players_xy, tr.players_xy, equal_nan=True)


def test_join_halves_keeps_halves_apart_for_the_scorer():
    arr = np.arange(10)
    out = harness._join_halves(arr, [(0, 4), (5, 9)], -1)
    assert len(out) == 10 + harness.SEPARATOR and (out[5:5 + harness.SEPARATOR] == -1).all()

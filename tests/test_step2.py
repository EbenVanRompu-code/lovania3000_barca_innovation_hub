import numpy as np

from possession import ball as ball_mod
from possession import deadball, evidence, hmm
from possession.params import Params
from possession.pipeline import decode_match
from synth import STANDARD, make_match


def test_long_runs_only_keeps_long_enough_runs():
    m = np.array([1, 1, 0, 1, 1, 1, 1, 0], dtype=bool)
    assert deadball.long_runs(m, 3).tolist() == [0, 0, 0, 1, 1, 1, 1, 0]


def test_missing_ball_is_not_dead_ball():
    tr, _ = make_match([("play", 0, 200, [(20, 150)])])
    b = ball_mod.clean(tr, Params())
    sig = deadball.signals(b, tr, Params())
    assert not sig["out"][30:160].any() and not sig["stationary"][30:160].any()


def test_evidence_sign_follows_the_closer_team():
    tr, _ = make_match([("play", 0, 100, []), ("play", 1, 100, [])])
    ev = evidence.variant_a(ball_mod.clean(tr, Params()), tr, Params())
    assert (ev[:100] > 0).all() and (ev[100:] < 0).all()
    assert np.abs(ev).max() <= Params().c


def test_evidence_is_zero_without_ball():
    tr, _ = make_match([("play", 0, 100, [(10, 60)])])
    ev = evidence.variant_a(ball_mod.clean(tr, Params()), tr, Params())
    assert (ev[10:70] == 0).all()


def test_viterbi_holds_state_through_flat_frames():
    ev = np.zeros(100)
    ev[:20] = 2.0
    path = hmm.decode(ev, np.zeros(100), Params())
    assert (path == hmm.H).all()


def test_merge_short_runs():
    path = np.array([0] * 50 + [1] * 5 + [0] * 50 + [2] * 40 + [1] * 60, dtype=np.int8)
    out = hmm.merge_short_runs(path, 25)
    assert (out[:105] == 0).all() and (out[105:145] == 2).all() and (out[145:] == 1).all()


def test_pipeline_recovers_scripted_possession():
    tr, truth = make_match(STANDARD, lead=20)
    n = tr.n_frames
    labels = decode_match(tr, [(20, n - 1)])
    assert (labels[:20] == "X").all() and (labels[20:] != "X").all()
    # compare away from the segment boundaries (the decoder needs a few frames to switch)
    ok = np.ones(n, dtype=bool)
    ok[:20] = False
    t0 = 20
    for kind, side, length, gaps in STANDARD:
        ok[t0:t0 + 40] = False      # first 40 frames of each segment
        t0 += length
    agree = (labels[ok] == truth[ok]).mean()
    assert agree > 0.97, agree


def test_forced_dead_block_is_all_d():
    tr, _ = make_match(STANDARD)
    labels = decode_match(tr, [(0, tr.n_frames - 1)], forced_dead=(50, 149))
    assert (labels[50:150] == "D").all()

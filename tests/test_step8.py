import numpy as np

from possession import ball as ball_mod
from possession import evidence
from possession.atd import Tracking
from possession.params import Params
from possession.pipeline import decode_match
from synth import STANDARD, make_match


def _track(ball_x, home_x, away_x):
    """1 home and 1 away player on the x axis; ball moves along it."""
    n = len(ball_x)
    ball = np.stack([ball_x, np.zeros(n)], axis=1)
    players = np.zeros((n, 2, 2))
    players[:, 0, 0] = home_x
    players[:, 1, 0] = away_x
    return Tracking(ball, np.full(n, np.nan), players, np.full((n, 2), np.nan),
                    np.array([0, 1], dtype=np.int8), 25)


def test_holder_switches_only_on_an_opposing_control_event():
    n = 120
    ball_x = np.zeros(n)
    home_x = np.zeros(n)
    away_x = np.full(n, 30.0)
    # frames 40-79: ball flies to the away player at 25 m/s (no control on the way)
    ball_x[40:80] = np.linspace(0.5, 30, 40)
    ball_x[80:] = 30.0
    tr = _track(ball_x, home_x, away_x)
    p = Params()
    holder = evidence.control_holder(ball_mod.clean(tr, p), tr, p)
    assert (holder[:40] == 1).all()          # home controls at the start
    assert (holder[45:75] == 1).all()        # flight: still home, no control event
    assert (holder[90:] == -1).all()         # away player gets it


def test_holder_survives_ball_gap():
    n = 100
    tr = _track(np.zeros(n), np.zeros(n), np.full(n, 30.0))
    tr.ball_xy[30:90] = np.nan
    p = Params()
    holder = evidence.control_holder(ball_mod.clean(tr, p), tr, p)
    assert (holder == 1).all()


def test_variant_b_recovers_scripted_possession():
    tr, truth = make_match(STANDARD, lead=0)
    labels = decode_match(tr, [(0, tr.n_frames - 1)], Params(), variant="B")
    t0, ok = 0, np.ones(tr.n_frames, dtype=bool)
    for kind, side, length, gaps in STANDARD:
        ok[t0:t0 + 40] = False
        t0 += length
    assert (labels[ok] == truth[ok]).mean() > 0.95

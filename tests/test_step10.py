import numpy as np

from possession import ball
from possession.params import Params


def _p(**kw):
    return Params(traj=True, **kw)


def test_prediction_follows_last_velocity_and_decays():
    xy = np.full((200, 2), np.nan)
    xy[:50] = np.stack([np.arange(50) * 0.2, np.full(50, 30.0)], axis=1)   # 0.2 m/frame to the right
    out, pred = ball.predict_long_gaps(xy, ~np.isnan(xy[:, 0]), _p(traj_max=40))
    assert pred[50:90].all() and not pred[90:].any()
    steps = np.diff(out[49:90, 0])
    assert steps[0] > 0.19 and (np.diff(steps) < 0).all()           # keeps moving, slowing down


def test_prediction_discarded_when_it_misses_the_next_sighting():
    xy = np.full((200, 2), np.nan)
    xy[:50] = np.stack([np.arange(50) * 0.2, np.full(50, 30.0)], axis=1)
    xy[150:] = [10.0, 5.0]                                           # reappears far away
    out, pred = ball.predict_long_gaps(xy, ~np.isnan(xy[:, 0]), _p())
    assert not pred.any() and np.isnan(out[60:140]).all()


def test_prediction_kept_when_it_agrees_with_the_next_sighting():
    f = 0.97
    xy = np.full((200, 2), np.nan)
    xy[:50] = np.stack([np.arange(50) * 0.2, np.full(50, 30.0)], axis=1)
    reach = (1 - f ** 101) / (1 - f)
    xy[150:] = [xy[49, 0] + 0.2 * reach, 30.0]
    out, pred = ball.predict_long_gaps(xy, ~np.isnan(xy[:, 0]), _p())
    assert pred[50:125].all()


def test_ball_leaving_the_pitch_is_marked_out():
    xy = np.full((200, 2), np.nan)
    xy[:50] = np.stack([100 + np.arange(50) * 0.3, np.full(50, 30.0)], axis=1)   # heading to the end line
    t = type("T", (), {"ball_xy": xy, "ball_speed": np.full(200, np.nan), "fps": 25})()
    b = ball.clean(t, _p())
    assert b.predicted.any() and b.out[50:].any()


def test_default_is_off():
    xy = np.full((100, 2), np.nan)
    xy[:20] = 1.0
    t = type("T", (), {"ball_xy": xy, "ball_speed": np.full(100, np.nan), "fps": 25})()
    assert not ball.clean(t, Params()).predicted.any()

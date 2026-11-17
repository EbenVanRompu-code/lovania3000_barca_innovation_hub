import numpy as np
import pandas as pd
import pytest

from possession import atd, ball
from possession.params import Params


def test_interpolation_fills_short_gap_only():
    xy = np.full((60, 2), np.nan)
    xy[0] = [0, 0]
    xy[5] = [10, 5]        # gap of 4 frames (1..4) -> filled
    xy[40] = [20, 5]       # gap of 34 frames (6..39) -> left NaN
    xy[41:] = [20, 5]
    out, filled, gap = ball.interpolate_gaps(xy, max_gap=25)
    assert np.allclose(out[1:5, 0], [2, 4, 6, 8]) and np.allclose(out[1:5, 1], [1, 2, 3, 4])
    assert filled[1:5].all() and not filled[6:40].any()
    assert np.isnan(out[6:40]).all()
    assert (gap[1:5] == 4).all() and (gap[6:40] == 34).all() and gap[0] == 0


def test_gap_at_the_edges_is_not_extrapolated():
    xy = np.full((10, 2), np.nan)
    xy[3:7] = 1.0
    out, filled, _ = ball.interpolate_gaps(xy, max_gap=25)
    assert np.isnan(out[:3]).all() and np.isnan(out[7:]).all() and not filled.any()


def test_speed_is_derived_from_positions_and_ignores_the_source():
    xy = np.zeros((9, 2))
    xy[:, 0] = np.arange(9) * 0.2          # 0.2 m per frame = 5 m/s at 25 fps
    provided = np.full(9, 99.0)            # a wrong source speed must be ignored
    v = ball.speed_of(xy, provided, 25, half=2)
    assert np.allclose(v, 5.0)


def test_speed_nan_where_position_nan():
    xy = np.zeros((7, 2))
    xy[3] = np.nan
    v = ball.speed_of(xy, None, 25)
    assert np.isnan(v[3])


def test_speed_smoothing_tames_jitter():
    rng = np.random.default_rng(0)
    xy = np.zeros((500, 2)) + rng.normal(0, 0.1, (500, 2))     # a still ball with 10 cm jitter
    raw = np.linalg.norm(np.diff(xy, axis=0), axis=1).mean() * 25
    assert ball.speed_of(xy, None, 25, half=2)[5:-5].mean() < raw / 2


def test_out_of_pitch_margin_and_nan():
    xy = np.array([[50, 30], [105.3 + 0.4, 30], [105.3 + 0.6, 30], [50, -0.6], [np.nan, np.nan]])
    assert ball.out_of_pitch(xy, margin=0.5).tolist() == [False, False, True, True, False]


def _frame(n=5):
    return pd.DataFrame({
        "frame_id": np.arange(2, 2 + n),       # video frames 1..n
        "ball_x": [0.5, np.nan, 0.5, 0.5, 0.5][:n], "ball_y": [0.5, np.nan, 0.5, 0.5, 0.5][:n],
        "ball_s": [1.0] * n,
        "1_x": [0.1] * n, "1_y": [0.2] * n, "1_s": [2.0] * n,
        "2_x": [0.9] * n, "2_y": [0.8] * n, "2_s": [3.0] * n,
    })


def test_from_frame_scales_aligns_and_assigns_teams():
    t = atd.from_frame(_frame(), {"1": 0, "2": 1}, n_frames=10)
    assert t.ball_xy.shape == (10, 2) and t.players_xy.shape == (10, 2, 2)
    assert np.isnan(t.ball_xy[0]).all()                     # frame_id 1 absent -> NaN
    assert np.allclose(t.ball_xy[1], [0.5 * 105.3, 0.5 * 68.0])   # frame_id 2 -> video frame 1
    assert np.isnan(t.ball_xy[2]).all()                     # NaN observation stays NaN
    assert np.allclose(t.players_xy[1, 1], [0.9 * 105.3, 0.8 * 68.0])
    assert t.team.tolist() == [0, 1]
    assert t.ball_speed[1] == 1.0 and t.players_speed[1, 1] == 3.0


def test_from_frame_rejects_unnormalised_coordinates():
    df = _frame()
    df["ball_x"] = 50.0
    with pytest.raises(ValueError):
        atd.from_frame(df, {"1": 0, "2": 1}, n_frames=10)


def test_from_frame_rejects_frame_id_out_of_range():
    with pytest.raises(ValueError):
        atd.from_frame(_frame(), {"1": 0, "2": 1}, n_frames=3)


def test_clean_runs_end_to_end():
    t = atd.from_frame(_frame(), {"1": 0, "2": 1}, n_frames=10)
    b = ball.clean(t, Params())
    assert b.filled[2] and b.observed[1] and not b.observed[2]

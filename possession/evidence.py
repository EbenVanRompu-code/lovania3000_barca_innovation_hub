"""Variant A: geometric evidence (PLAN.md section 3).

evidence = clip((d_away - d_home) / s, -c, +c) * exp(-ball_speed / v0)

d_home / d_away: distance (m) from the ball to the nearest home / away player. Positive evidence
favours H. A fast ball gets a small weight. Frames with no ball position, or with no player of one
team tracked, get 0 (flat emission), so the decoder carries its state through them.
"""

import numpy as np

from .params import Params

HOME, AWAY = 0, 1


def nearest_distance(ball_xy, players_xy, team, side):
    """(T,) distance in metres to the nearest tracked player of `side`; NaN if none / no ball."""
    pos = players_xy[:, team == side, :]
    d = np.linalg.norm(pos - ball_xy[:, None, :], axis=2)
    d = np.where(np.isnan(d), np.inf, d)
    nearest = d.min(axis=1) if d.shape[1] else np.full(len(ball_xy), np.inf)
    nearest[~np.isfinite(nearest)] = np.nan
    nearest[np.isnan(ball_xy).any(axis=1)] = np.nan
    return nearest


def variant_a(ball, tracking, params=None):
    p = params or Params()
    d_home = nearest_distance(ball.xy, tracking.players_xy, tracking.team, HOME)
    d_away = nearest_distance(ball.xy, tracking.players_xy, tracking.team, AWAY)
    raw = np.clip((d_away - d_home) / p.s, -p.c, p.c)
    weight = np.exp(-np.nan_to_num(ball.speed, nan=0.0) / p.v0)
    ev = raw * weight
    return np.nan_to_num(ev, nan=0.0)


def control_holder(ball, tracking, p):
    """(T,) +1 / -1 / 0: the team of the last control event, 0 before the first one.

    Control = nearest player of a team within d_control of the ball, ball slower than v_control,
    for at least k_control frames in a row. If both teams are within d_control the nearer one
    controls. A pass or deflection without control never changes the holder.
    """
    from .deadball import long_runs
    d_home = nearest_distance(ball.xy, tracking.players_xy, tracking.team, HOME)
    d_away = nearest_distance(ball.xy, tracking.players_xy, tracking.team, AWAY)
    slow = np.nan_to_num(ball.speed, nan=0.0) < p.v_control   # unknown speed does not block control
    with np.errstate(invalid="ignore"):
        home = (d_home < p.d_control) & slow & ~(d_away < d_home)
        away = (d_away < p.d_control) & slow & ~(d_home <= d_away)
    n = len(home)
    event_team = np.zeros(n, dtype=np.int8)
    for team_mask, sign in ((home, 1), (away, -1)):
        run = long_runs(team_mask, p.k_control)
        starts = run & ~np.concatenate(([False], run[:-1]))
        event_team[starts] = sign
    idx = np.where(event_team != 0, np.arange(n), -1)
    last = np.maximum.accumulate(idx)
    return np.where(last >= 0, event_team[np.maximum(last, 0)], 0)


def variant_b(ball, tracking, params=None):
    p = params or Params()
    ev = p.b_strength * control_holder(ball, tracking, p).astype(float)
    if p.b_mix:
        ev = ev + p.b_mix * variant_a(ball, tracking, p)
    return ev


def _variant_b(ball, tracking, params):
    return variant_b(ball, tracking, params), 0.0


def _variant_a(ball, tracking, params):
    """Registry signature: returns (H/A evidence, extra D score)."""
    return variant_a(ball, tracking, params), 0.0


def _variant_c(ball, tracking, params):
    from . import classifier
    return classifier.predict_evidence(ball, tracking, params)


VARIANTS = {"A": _variant_a, "B": _variant_b, "C": _variant_c}
# Variant C learns D itself, so the hand-written dead-ball rules are not added on top of it.
NO_RULES = {"C"}

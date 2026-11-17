"""Dead-ball signals (PLAN.md section 4). They are scores added to the D emission, never hard rules,
except the fixed pre-kickoff block which the organisers published (FAQ.md).

Rule 1  ball outside the pitch (after interpolation)
Rule 2  ball stationary: speed < stationary_speed for at least stationary_min_frames in a row
Rule 3  players slow and static: mean tracked-player speed < players_slow_speed for at least
        players_slow_min_frames in a row (weak signal, works without a ball)
Rule 4  fixed pre-kickoff block (hard)
A missing ball is never dead ball: rules 1 and 2 need a ball position, and a NaN speed breaks a run.
"""

import numpy as np

from .params import Params

HARD_D = 1e4  # emission score that forces D (rule 4)


def long_runs(mask, min_frames):
    """True only inside runs of True that are at least min_frames long."""
    out = np.zeros(len(mask), dtype=bool)
    padded = np.concatenate(([False], mask, [False])).astype(np.int8)
    d = np.diff(padded)
    for s, e in zip(np.flatnonzero(d == 1), np.flatnonzero(d == -1)):
        if e - s >= min_frames:
            out[s:e] = True
    return out


def players_mean_speed(players_xy, players_speed, fps, half=2):
    """Mean speed (m/s) over tracked players per frame, from positions (source speeds ignored).
    NaN when no player is tracked."""
    from .ball import smooth_speed
    speed = smooth_speed(players_xy, fps, half)
    count = np.isfinite(speed).sum(axis=1)
    total = np.where(np.isfinite(speed), speed, 0.0).sum(axis=1)
    return np.where(count > 0, total / np.maximum(count, 1), np.nan)


def signals(ball, tracking, params=None):
    """dict of boolean arrays (T,): out, stationary, players_slow."""
    p = params or Params()
    with np.errstate(invalid="ignore"):
        stationary = long_runs(ball.speed < p.stationary_speed, p.stationary_min_frames)
        mean_speed = players_mean_speed(tracking.players_xy, tracking.players_speed, tracking.fps,
                                        p.speed_half_window)
        slow = long_runs(mean_speed < p.players_slow_speed, p.players_slow_min_frames)
    return {"out": ball.out, "stationary": stationary, "players_slow": slow}


def combine(sig, params, forced=None):
    """D score from precomputed signals (so tuning weights does not recompute the signals)."""
    p = params
    score = (p.w_out * sig["out"] + p.w_stationary * sig["stationary"]
             + p.w_players * sig["players_slow"]).astype(float)
    if forced is not None:
        score[forced[0]:forced[1] + 1] = HARD_D
    return score


def dead_score(ball, tracking, params=None, forced=None):
    """Per-frame score added to the D emission. `forced` = (first, last) video frames that are
    always D (rule 4); applied as a hard score."""
    p = params or Params()
    return combine(signals(ball, tracking, p), p, forced)

"""Ball cleaning: short-gap interpolation, speed, out-of-pitch (PLAN.md section 4).

A missing ball is never treated as dead ball here. Gaps longer than `max_gap` stay NaN and the
decoder carries its state through them.
"""

from dataclasses import dataclass

import numpy as np

from .params import PITCH, Params


@dataclass
class Ball:
    xy: np.ndarray        # (T, 2) metres; NaN in gaps longer than max_gap
    speed: np.ndarray     # (T,) m/s; NaN where xy is NaN
    observed: np.ndarray  # (T,) bool, ball seen by the source on this frame
    filled: np.ndarray    # (T,) bool, position came from interpolation
    gap_len: np.ndarray   # (T,) int, length of the unobserved run this frame sits in (0 = seen)
    out: np.ndarray       # (T,) bool, ball outside the pitch + margin (False where xy is NaN)
    predicted: np.ndarray = None   # (T,) bool, position came from trajectory reconstruction


def _runs(mask):
    """(start, end_exclusive) of each True run."""
    padded = np.concatenate(([False], mask, [False])).astype(np.int8)
    d = np.diff(padded)
    return np.flatnonzero(d == 1), np.flatnonzero(d == -1)


def interpolate_gaps(xy, max_gap):
    """Linear interpolation across NaN runs of at most max_gap frames, with a seen frame on both
    sides. Returns (xy_filled, filled mask, gap_len per frame)."""
    xy = xy.copy()
    missing = np.isnan(xy).any(axis=1)
    filled = np.zeros(len(xy), dtype=bool)
    gap_len = np.zeros(len(xy), dtype=np.int64)
    for start, end in zip(*_runs(missing)):
        n = end - start
        gap_len[start:end] = n
        if n <= max_gap and start > 0 and end < len(xy):
            for k in range(2):
                xy[start:end, k] = np.interp(
                    np.arange(start, end), [start - 1, end], [xy[start - 1, k], xy[end, k]])
            filled[start:end] = True
    return xy, filled, gap_len


def predict_long_gaps(xy, observed, p):
    """Fill the start of gaps longer than max_gap by extrapolating the last velocity with friction.

    Trusted for at most traj_max frames and only if, when the ball is seen again, the prediction
    (carried to that moment) lands within traj_tol metres of it; otherwise the whole gap is left
    empty. Returns (xy, predicted mask). There is no ball height in the data, so this only makes
    sense for ground-like motion.
    """
    xy = xy.copy()
    predicted = np.zeros(len(xy), dtype=bool)
    missing = np.isnan(xy).any(axis=1)
    f = p.traj_decay
    for start, end in zip(*_runs(missing)):
        if end - start <= p.max_gap or start < 6:
            continue
        v = (xy[start - 1] - xy[start - 6]) / 5.0           # m per frame over the last 5 frames
        if np.isnan(v).any():
            continue
        steps = np.arange(1, end - start + 1)
        reach = (1 - f ** steps) / (1 - f)                  # sum of decaying steps
        path = xy[start - 1] + reach[:, None] * v
        if end < len(xy) and not np.isnan(xy[end]).any():
            if np.linalg.norm(path[-1] - xy[end]) > p.traj_tol:
                continue
        n = min(len(steps), p.traj_max)
        xy[start:start + n] = path[:n]
        predicted[start:start + n] = True
    return xy, predicted


def smooth_speed(xy, fps, half):
    """Speed (m/s) as the central difference over +/-half frames: |xy[t+half] - xy[t-half]| /
    (2*half) * fps. NaN where either end is missing; the first/last `half` frames use the
    one-sided difference. Works for (T, 2) and (T, N, 2) arrays (speed per object)."""
    n = xy.shape[0]
    out = np.full(xy.shape[:-1], np.nan)
    if half < 1 or n < 2:
        half = 1
    idx = np.arange(n)
    lo, hi = np.maximum(idx - half, 0), np.minimum(idx + half, n - 1)
    span = (hi - lo).astype(float)
    d = np.linalg.norm(xy[hi] - xy[lo], axis=-1)
    span = span.reshape((n,) + (1,) * (d.ndim - 1))
    with np.errstate(invalid="ignore", divide="ignore"):
        out = np.where(span > 0, d / span * fps, np.nan)
    out[np.isnan(xy).any(axis=-1)] = np.nan      # no speed where the object itself is missing
    return out


def speed_of(xy, provided, fps, half=2):
    """Ball speed in m/s from positions (the source's own speed is ignored on purpose)."""
    return smooth_speed(xy, fps, half)


def out_of_pitch(xy, pitch=PITCH, margin=0.5):
    length, width = pitch
    x, y = xy[:, 0], xy[:, 1]
    with np.errstate(invalid="ignore"):
        out = (x < -margin) | (x > length + margin) | (y < -margin) | (y > width + margin)
    return out & ~np.isnan(xy).any(axis=1)


def clean(tracking, params=None, pitch=PITCH):
    p = params or Params()
    observed = ~np.isnan(tracking.ball_xy).any(axis=1)
    xy, filled, gap_len = interpolate_gaps(tracking.ball_xy, p.max_gap)
    predicted = np.zeros(len(xy), dtype=bool)
    if p.traj:
        xy, predicted = predict_long_gaps(xy, observed, p)
    return Ball(
        xy=xy,
        speed=speed_of(xy, None, tracking.fps, p.speed_half_window),
        observed=observed,
        filled=filled,
        gap_len=gap_len,
        out=out_of_pitch(xy, pitch, p.pitch_margin),
        predicted=predicted,
    )

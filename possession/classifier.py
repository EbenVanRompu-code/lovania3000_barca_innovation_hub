"""Variant C: learned evidence (PLAN.md section 3).

A gradient-boosted tree classifier over H / A / D, trained on DFL tracking that has been degraded
to look like ours. It uses sklearn's HistGradientBoostingClassifier: the same model family as
LightGBM, but with no system library to install (LightGBM needs libomp on macOS). Seeded and
deterministic. Probabilities are calibrated by a single temperature fitted on held-out matches.

The model never sees our match. Its output goes into the same decoder as variants A and B:
    evidence = clip(log pH - log pA)           (>0 favours home)
    D score  = log pD - (log pH + log pA) / 2
Features avoid pitch direction (teams swap ends at half time), so they transfer across matches.
"""

import numpy as np
from scipy.optimize import minimize_scalar
from sklearn.ensemble import HistGradientBoostingClassifier

from .params import PITCH, Params

FEATURES = []   # filled by compute_features (names, in order)
ACTIVE = None   # the model used by VARIANTS["C"]; set by the harness / the final run


def roll_mean(x, width, direction):
    """NaN-aware mean over the `width` frames before (-1) or after (+1) each frame, excluding it."""
    ok = np.isfinite(x)
    v = np.where(ok, x, 0.0)
    cs = np.concatenate(([0.0], np.cumsum(v)))
    cn = np.concatenate(([0], np.cumsum(ok)))
    n = len(x)
    idx = np.arange(n)
    if direction < 0:
        lo, hi = np.maximum(idx - width, 0), idx
    else:
        lo, hi = np.minimum(idx + 1, n), np.minimum(idx + 1 + width, n)
    cnt = cn[hi] - cn[lo]
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(cnt > 0, (cs[hi] - cs[lo]) / np.maximum(cnt, 1), np.nan)


def _two_nearest(ball_xy, players_xy, team, side):
    pos = players_xy[:, team == side, :]
    d = np.linalg.norm(pos - ball_xy[:, None, :], axis=2)
    d = np.where(np.isnan(d), np.inf, d)
    if d.shape[1] < 2:
        return np.full(len(ball_xy), np.nan), np.full(len(ball_xy), np.nan)
    part = np.partition(d, 1, axis=1)[:, :2]
    part.sort(axis=1)
    part = np.where(np.isfinite(part), part, np.nan)
    part[np.isnan(ball_xy).any(axis=1)] = np.nan
    return part[:, 0], part[:, 1]


def _player_speeds(tracking, half=2):
    from .ball import smooth_speed
    return smooth_speed(tracking.players_xy, tracking.fps, half)


def compute_features(ball, tracking, params=None, pitch=PITCH):
    """(T, F) float matrix; NaN means unknown (the model handles it natively)."""
    p = params or Params()
    length, width = pitch
    h1, h2 = _two_nearest(ball.xy, tracking.players_xy, tracking.team, 0)
    a1, a2 = _two_nearest(ball.xy, tracking.players_xy, tracking.team, 1)
    diff = a1 - h1
    speed = ball.speed

    # heading from a 5-frame displacement (robust to position jitter); turn = change over 5 frames
    v = np.full_like(ball.xy, np.nan)
    v[2:-2] = ball.xy[4:] - ball.xy[:-4]
    heading = np.arctan2(v[:, 1], v[:, 0])
    turn = np.full(len(heading), np.nan)
    turn[5:] = np.abs(np.angle(np.exp(1j * (heading[5:] - heading[:-5]))))

    x, y = ball.xy[:, 0], ball.xy[:, 1]
    line = np.minimum.reduce([x, length - x, y, width - y])

    ps = _player_speeds(tracking, p.speed_half_window)
    mean_ps = _mean_rows(ps)
    walking = np.sum(ps < 1.5, axis=1).astype(float)

    team = tracking.team
    cols = {"d_home1": h1, "d_home2": h2, "d_away1": a1, "d_away2": a2, "d_diff": diff,
            "ball_speed": speed, "ball_turn": turn, "ball_out": ball.out.astype(float),
            "ball_line": line, "mean_player_speed": mean_ps, "n_walking": walking,
            "n_home": np.isfinite(tracking.players_xy[:, team == 0, 0]).sum(1).astype(float),
            "n_away": np.isfinite(tracking.players_xy[:, team == 1, 0]).sum(1).astype(float),
            "gap_len": np.minimum(ball.gap_len, 500).astype(float),
            "observed": ball.observed.astype(float)}
    for side, name in ((0, "home"), (1, "away")):
        pos = tracking.players_xy[:, team == side, :]
        cols[name + "_spread"] = _spread(pos)
    cols["centroid_dist"] = np.linalg.norm(_centroid(tracking.players_xy[:, team == 0, :])
                                           - _centroid(tracking.players_xy[:, team == 1, :]), axis=1)
    for src_name, src in (("d_diff", diff), ("ball_speed", speed), ("mean_player_speed", mean_ps),
                          ("observed", ball.observed.astype(float))):
        for w in (25, 75):
            cols["{}_past{}".format(src_name, w)] = roll_mean(src, w, -1)
            cols["{}_fut{}".format(src_name, w)] = roll_mean(src, w, +1)
    FEATURES[:] = list(cols)
    return np.column_stack([cols[k] for k in FEATURES])


def _mean_rows(a):
    ok = np.isfinite(a)
    cnt = ok.sum(axis=1)
    return np.where(cnt > 0, np.where(ok, a, 0.0).sum(axis=1) / np.maximum(cnt, 1), np.nan)


def _centroid(pos):
    ok = np.isfinite(pos[..., 0])
    cnt = ok.sum(axis=1)
    s = np.where(ok[..., None], pos, 0.0).sum(axis=1)
    return np.where(cnt[:, None] > 0, s / np.maximum(cnt, 1)[:, None], np.nan)


def _spread(pos):
    c = _centroid(pos)
    d = np.linalg.norm(pos - c[:, None, :], axis=2)
    return _mean_rows(d)


class Model:
    def __init__(self, clf, temperature):
        self.clf = clf
        self.temperature = temperature

    def proba(self, X):
        logp = np.log(np.clip(self.clf.predict_proba(X), 1e-6, 1.0))
        z = logp / self.temperature
        z -= z.max(axis=1, keepdims=True)
        e = np.exp(z)
        return e / e.sum(axis=1, keepdims=True)


def fit(X, y, seed=0, max_iter=150):
    clf = HistGradientBoostingClassifier(max_iter=max_iter, learning_rate=0.1, max_leaf_nodes=31,
                                         l2_regularization=1.0, random_state=seed)
    clf.fit(X, y)
    return clf


def fit_temperature(clf, X, y):
    """One scalar T minimising log-loss of softmax(log p / T) on held-out data."""
    logp = np.log(np.clip(clf.predict_proba(X), 1e-6, 1.0))

    def nll(t):
        z = logp / t
        z = z - z.max(axis=1, keepdims=True)
        lp = z - np.log(np.exp(z).sum(axis=1, keepdims=True))
        return -lp[np.arange(len(y)), y].mean()

    return float(minimize_scalar(nll, bounds=(0.3, 5.0), method="bounded").x)


def evidence_from_proba(proba, params=None):
    p = params or Params()
    lp = np.log(np.clip(proba, 1e-6, 1.0))
    ev = np.clip((lp[:, 0] - lp[:, 1]), -p.c_model, p.c_model)
    dead = lp[:, 2] - 0.5 * (lp[:, 0] + lp[:, 1])
    return ev, p.d_model * dead


def predict_evidence(ball, tracking, params=None):
    """Registry signature for VARIANTS["C"]: (evidence, D score). Uses the ACTIVE model."""
    if ACTIVE is None:
        raise RuntimeError("variant C needs classifier.ACTIVE to be set (a trained Model)")
    X = compute_features(ball, tracking, params)
    return evidence_from_proba(ACTIVE.proba(X), params)

"""A scripted fake match for tests: known possession, ball gaps, a ball out of play, a still ball.

Segments are (kind, team, n_frames, gaps). kind is "play" (team = 0 home / 1 away carries the ball),
"out" (ball outside the pitch, players keep jogging) or "still" (ball stationary, players
walking). gaps = [(offset, length)] frames of missing ball inside the segment.
"""

import numpy as np

from possession.atd import Tracking

FPS = 25
PITCH = (105.3, 68.0)
N_SLOTS = 46          # as in the real data: 22 used, the rest always empty


def make_match(segments, seed=0, lead=0):
    rng = np.random.default_rng(seed)
    n = lead + sum(s[2] for s in segments)
    ball = np.full((n, 2), np.nan)
    players = np.full((n, N_SLOTS, 2), np.nan)
    truth = np.full(n, "X", dtype="<U1")
    team = np.array([i % 2 for i in range(N_SLOTS)], dtype=np.int8)
    used = {0: [i for i in range(22) if team[i] == 0][:11], 1: [i for i in range(22) if team[i] == 1][:11]}
    # i % 2 gives 11 slots per side among the first 22
    theta0 = rng.uniform(0, 2 * np.pi, N_SLOTS)
    radius = rng.uniform(8, 14, N_SLOTS)
    centre = np.array([PITCH[0] / 2, PITCH[1] / 2])

    t0 = lead
    for kind, side, length, gaps in segments:
        t = np.arange(length)
        if kind == "play":
            # ball drifts at ~3 m/s around the centre
            b = centre + np.stack([20 * np.sin(0.0048 * t), 10 * np.sin(0.0096 * t)], axis=1) * 1.0
            omega = 0.25 / FPS
            truth[t0:t0 + length] = "HA"[side]
        elif kind == "out":
            b = np.stack([np.full(length, -5.0), 30 + 0.12 * t], axis=1)   # beyond the touchline
            omega = 0.25 / FPS
            truth[t0:t0 + length] = "D"
        elif kind == "still":
            b = np.tile(centre, (length, 1))
            omega = 0.03 / FPS
            truth[t0:t0 + length] = "D"
        else:
            raise ValueError(kind)

        anchor = b if kind == "play" else np.tile(centre, (length, 1))
        for slot in range(22):
            ang = theta0[slot] + omega * t
            r = radius[slot]
            if kind == "play":
                # teammates of the carrier stay 8..14 m away, opponents 6..14 m away
                r = r if team[slot] == side else r - 2
            players[t0:t0 + length, slot, 0] = anchor[:, 0] + r * np.cos(ang)
            players[t0:t0 + length, slot, 1] = anchor[:, 1] + r * np.sin(ang)
        if kind == "play":
            carrier = used[side][0]
            players[t0:t0 + length, carrier] = b + np.array([1.0, 0.0])   # 1 m from the ball
        ball[t0:t0 + length] = b
        for offset, glen in gaps:
            ball[t0 + offset:t0 + offset + glen] = np.nan
        t0 += length

    return Tracking(
        ball_xy=ball,
        ball_speed=np.full(n, np.nan),
        players_xy=players,
        players_speed=np.full((n, N_SLOTS), np.nan),
        team=team,
        fps=FPS,
    ), truth


STANDARD = [
    ("play", 0, 200, []),
    ("play", 1, 300, [(60, 15), (150, 80)]),   # short gap (interpolated) and long gap (carried)
    ("out", 0, 100, []),
    ("play", 0, 200, []),
    ("still", 0, 150, []),
    ("play", 1, 200, []),
]

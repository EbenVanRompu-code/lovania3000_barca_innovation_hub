"""Make clean DFL tracking look like our noisy ATD (PLAN.md section 5, step 5).

Only DOCUMENTED facts about our data are used: the ball is missing on ~41% of graded frames
(DATA.md) and there are fragmented, duplicated tracks. We did NOT measure gap lengths on our match,
so the gap length is swept over a range of profiles and settings are preferred that hold up across
all of them.

  ball:    alternating seen / missing runs, geometric lengths, mean missing run = `mean_gap`,
           mean seen run set so the missing share is `missing_share` (default 0.41)
  jitter:  gaussian position noise on ball and players (reasoning, untuned values)
  players: each slot drops out in short runs (fragmented tracks)
"""

from dataclasses import dataclass, replace

import numpy as np


@dataclass(frozen=True)
class Profile:
    name: str
    mean_gap: int = 50           # frames, mean length of a ball-missing run
    missing_share: float = 0.41  # DATA.md
    ball_jitter: float = 0.2     # m, reasoning
    player_jitter: float = 0.1   # m, reasoning
    player_missing: float = 0.04    # share of player-slot frames dropped
    player_gap: int = 40            # frames, mean length of a dropped run


# gap-length sweep: short flickers ... long occlusions. None of these is "our" gap profile.
PROFILES = {
    "gap10": Profile("gap10", mean_gap=10),
    "gap25": Profile("gap25", mean_gap=25),
    "gap75": Profile("gap75", mean_gap=75),
    "gap200": Profile("gap200", mean_gap=200),
}
# jitter sweep: we have NOT measured our tracking noise, and speeds are derived from positions, so
# the model is trained and tested over a range of position-noise levels (ball, players in metres)
NOISE_PROFILES = {
    "gap25_j05": Profile("gap25_j05", mean_gap=25, ball_jitter=0.10, player_jitter=0.05),
    "gap75_j05": Profile("gap75_j05", mean_gap=75, ball_jitter=0.10, player_jitter=0.05),
    "gap75_j2": Profile("gap75_j2", mean_gap=75, ball_jitter=0.40, player_jitter=0.20),
    "gap200_j2": Profile("gap200_j2", mean_gap=200, ball_jitter=0.40, player_jitter=0.20),
}
ALL_PROFILES = {**PROFILES, **NOISE_PROFILES}
CLEAN = Profile("clean", mean_gap=0, missing_share=0.0, ball_jitter=0.0, player_jitter=0.0,
                player_missing=0.0)


def _missing_mask(n, share, mean_gap, rng):
    """Boolean mask: True = missing. Alternating geometric runs with the requested share."""
    if share <= 0 or mean_gap <= 0:
        return np.zeros(n, dtype=bool)
    mean_seen = mean_gap * (1 - share) / share
    mask = np.zeros(n, dtype=bool)
    i = 0
    missing = rng.random() < share
    while i < n:
        length = int(rng.geometric(1.0 / (mean_gap if missing else mean_seen)))
        if missing:
            mask[i:i + length] = True
        i += length
        missing = not missing
    return mask


def degrade(tracking, blocks, profile, seed=0):
    """A degraded copy. The gap process runs inside each half block separately."""
    rng = np.random.default_rng(seed)
    t = replace(tracking, ball_xy=tracking.ball_xy.copy(), players_xy=tracking.players_xy.copy(),
                ball_speed=tracking.ball_speed.copy(), players_speed=tracking.players_speed.copy())
    for first, last in blocks:
        n = last - first + 1
        miss = _missing_mask(n, profile.missing_share, profile.mean_gap, rng)
        t.ball_xy[first:last + 1][miss] = np.nan
        t.ball_speed[first:last + 1][miss] = np.nan
        for slot in range(t.players_xy.shape[1]):
            pm = _missing_mask(n, profile.player_missing, profile.player_gap, rng)
            t.players_xy[first:last + 1, slot][pm] = np.nan
            t.players_speed[first:last + 1, slot][pm] = np.nan
    if profile.ball_jitter:
        t.ball_xy += rng.normal(0, profile.ball_jitter, t.ball_xy.shape)
    if profile.player_jitter:
        t.players_xy += rng.normal(0, profile.player_jitter, t.players_xy.shape)
    return t

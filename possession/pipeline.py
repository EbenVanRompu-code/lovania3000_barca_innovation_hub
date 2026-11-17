"""Tracking -> labels for every video frame. H/A/D inside the half blocks, X outside, never X inside."""

import numpy as np

from . import ball as ball_mod
from . import deadball, evidence, hmm
from .params import Params


def decode_match(tracking, blocks, params=None, forced_dead=None, variant="A", tag_factors=None):
    """blocks: [(first_frame, last_frame)] inclusive, in video frames.
    forced_dead: (first, last) frames that are always D, or None.
    Returns an array of 'H'/'A'/'D'/'X' strings of length tracking.n_frames."""
    p = params or Params()
    ball = ball_mod.clean(tracking, p)
    ev, extra_dead = evidence.VARIANTS[variant](ball, tracking, p)
    if variant in evidence.NO_RULES:
        dead = np.zeros(tracking.n_frames) + extra_dead
        if forced_dead is not None:
            dead[forced_dead[0]:forced_dead[1] + 1] = deadball.HARD_D
    else:
        dead = deadball.dead_score(ball, tracking, p, forced_dead) + extra_dead
    if tag_factors is not None:      # Smart Tagging (tagging.py): (ev_add, dead_add)
        ev = ev + tag_factors[0]
        dead = dead + tag_factors[1]
    labels = np.full(tracking.n_frames, "X", dtype="<U1")
    for first, last in blocks:
        sl = slice(first, last + 1)
        labels[sl] = hmm.LETTERS[hmm.decode(ev[sl], dead[sl], p)]
    return labels

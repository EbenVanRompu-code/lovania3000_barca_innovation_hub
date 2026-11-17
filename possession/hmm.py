"""Shared decoder: Viterbi over H / A / D per half, then a min_hold post-filter (PLAN.md section 3).

Emission log-scores per frame: H = +ev/2, A = -ev/2, D = dead_score. A frame with no evidence has
all three at 0 (flat), so the state carries through. Staying is free; H<->A costs switch_penalty;
entering or leaving D costs dead_penalty.
"""

import numpy as np

from .params import Params

H, A, D = 0, 1, 2
LETTERS = np.array(["H", "A", "D"])


def emissions(evidence, dead):
    return np.stack([0.5 * evidence, -0.5 * evidence, dead], axis=1)


def transition_costs(p):
    t = np.zeros((3, 3))
    t[H, A] = t[A, H] = p.switch_penalty
    t[H, D] = t[D, H] = t[A, D] = t[D, A] = p.dead_penalty
    return t


def viterbi(em, trans):
    """Best H/A/D path. Unrolled for 3 states on plain Python floats: about 10x faster than numpy
    for this small-state, long-sequence loop, which matters when tuning runs it thousands of times."""
    sw, dp = float(trans[H, A]), float(trans[H, D])
    n = len(em)
    rows = em.tolist()
    d0, d1, d2 = rows[0]
    back = bytearray(3 * n)
    for t in range(1, n):
        e0, e1, e2 = rows[t]
        # best predecessor for each state; ties prefer staying, then H, then A
        a, b, c = d0, d1 - sw, d2 - dp
        if a >= b and a >= c:
            n0, p0 = a, 0
        elif b >= c:
            n0, p0 = b, 1
        else:
            n0, p0 = c, 2
        a, b, c = d0 - sw, d1, d2 - dp
        if b >= a and b >= c:
            n1, p1 = b, 1
        elif a >= c:
            n1, p1 = a, 0
        else:
            n1, p1 = c, 2
        a, b, c = d0 - dp, d1 - dp, d2
        if c >= a and c >= b:
            n2, p2 = c, 2
        elif a >= b:
            n2, p2 = a, 0
        else:
            n2, p2 = b, 1
        back[3 * t], back[3 * t + 1], back[3 * t + 2] = p0, p1, p2
        d0, d1, d2 = n0 + e0, n1 + e1, n2 + e2
    path = np.empty(n, dtype=np.int8)
    state = 0 if (d0 >= d1 and d0 >= d2) else (1 if d1 >= d2 else 2)
    path[-1] = state
    for t in range(n - 1, 0, -1):
        state = back[3 * t + state]
        path[t - 1] = state
    return path


def merge_short_runs(path, min_hold):
    """Merge each H/A run shorter than min_hold into its longer neighbour (previous on a tie).
    Shortest first, repeated until none is left. D runs are never merged away."""
    runs = []   # [label, length]
    for lab in path:
        if runs and runs[-1][0] == lab:
            runs[-1][1] += 1
        else:
            runs.append([int(lab), 1])
    while len(runs) > 1:
        short = [(r[1], i) for i, r in enumerate(runs) if r[0] != D and r[1] < min_hold]
        if not short:
            break
        _, i = min(short)
        prev_len = runs[i - 1][1] if i > 0 else -1
        next_len = runs[i + 1][1] if i + 1 < len(runs) else -1
        j = i - 1 if prev_len >= next_len else i + 1
        runs[i][0] = runs[j][0]
        merged = []
        for r in runs:
            if merged and merged[-1][0] == r[0]:
                merged[-1][1] += r[1]
            else:
                merged.append(r)
        runs = merged
    return np.concatenate([np.full(n, lab, dtype=np.int8) for lab, n in runs])


def decode(evidence, dead, params=None):
    """Codes 0/1/2 (H/A/D) for one contiguous block of frames."""
    p = params or Params()
    path = viterbi(emissions(evidence, dead), transition_costs(p))
    return merge_short_runs(path, p.min_hold)

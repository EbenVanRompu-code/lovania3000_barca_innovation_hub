"""Grid tuning of variant A's decoder on DFL (PLAN.md step 6). Protocol, fixed before looking at scores:

  * tune on GRID_MATCHES (3), confirm on CONFIRM_MATCHES (4), never on our match or its GT
  * objective = mean spellF1 over matches and over the four degraded gap profiles
  * also report the worst profile; prefer the centre of a flat region (selection rule, PLAN.md s6)

    uv run python -m possession.tune stage1
"""

import csv
import itertools
import os
import sys
from dataclasses import replace

import numpy as np

from . import harness
from .external.degrade import PROFILES
from .params import Params

OUT = os.path.join(os.path.dirname(harness.RUNLOG))


def grid(base, axes, matches, tag, out_name):
    names = list(axes)
    rows = []
    for combo in itertools.product(*[axes[n] for n in names]):
        p = replace(base, **dict(zip(names, combo)))
        per_profile = []
        for prof in PROFILES:
            r = harness.evaluate_fast(p, matches, prof)
            per_profile.append(harness.mean(r))
        rows.append({**dict(zip(names, combo)), "mean_spellF1": float(np.mean(per_profile)),
                     "worst_profile": float(np.min(per_profile)),
                     **{"spell_" + k: v for k, v in zip(PROFILES, per_profile)}})
    os.makedirs(OUT, exist_ok=True)
    with open(os.path.join(OUT, out_name), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    return rows


STAGE1 = {"switch_penalty": [1, 2, 4, 8, 16], "dead_penalty": [1, 2, 5, 10],
          "min_hold": [1, 5, 12, 25, 50]}

def sweep(base, name, values, matches, variant="A"):
    """One-parameter-at-a-time sweep: [(value, mean spellF1 over profiles, worst profile)]."""
    out = []
    for v in values:
        p = replace(base, **{name: v})
        per = [harness.mean(harness.evaluate_fast(p, matches, prof, variant)) for prof in PROFILES]
        out.append((v, float(np.mean(per)), float(np.min(per))))
    return out


# stage 2: signal and evidence parameters, one at a time around the stage-1 choice
STAGE2 = {"s": [1, 2, 4, 8], "c": [1.5, 3, 6], "v0": [5, 10, 20, 40],
          "w_out": [2, 6, 12], "w_stationary": [1, 4, 8], "w_players": [0, 1, 2, 4, 8],
          "players_slow_speed": [0.8, 1.2, 1.8], "stationary_min_frames": [25, 50, 100]}

# stage 1 put the best cells on the grid edge (switch_penalty 1, min_hold 50), so the grid is
# extended in that direction (decided after seeing stage 1, on the same 3 grid matches)
STAGE1B = {"switch_penalty": [0.25, 0.5, 1, 2], "dead_penalty": [0.5, 1, 2, 5],
           "min_hold": [50, 75, 100, 150, 250]}

if __name__ == "__main__":
    import json
    stage = sys.argv[1]
    if stage == "stage2":
        base = replace(Params(), **json.load(open(sys.argv[2])))
        for name, values in STAGE2.items():
            for v, m, w in sweep(base, name, values, harness.GRID_MATCHES):
                print("{:<24s} {:>8} mean {:.4f} worst {:.4f}".format(name, v, m, w), flush=True)
    if stage == "stage1b":
        rows = grid(Params(), STAGE1B, harness.GRID_MATCHES, "stage1b", "grid_stage1b.csv")
        rows.sort(key=lambda r: -r["mean_spellF1"])
        for r in rows[:15]:
            print({k: (round(v, 4) if isinstance(v, float) else v) for k, v in r.items()})
    if stage == "stage1":
        rows = grid(Params(), STAGE1, harness.GRID_MATCHES, "stage1", "grid_stage1.csv")
        rows.sort(key=lambda r: -r["mean_spellF1"])
        for r in rows[:15]:
            print({k: (round(v, 4) if isinstance(v, float) else v) for k, v in r.items()})

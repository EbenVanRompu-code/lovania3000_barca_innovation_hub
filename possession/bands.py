"""Plausibility bands from the 7 DFL matches (PLAN.md section 6).

Computed from OUR PIPELINE'S OUTPUT on degraded DFL (apples to apples with what run.py produces on
our match), min-max over matches and gap profiles plus a relative margin. A band violation is a
veto to investigate, never a target to tune toward. Written to possession/bands_dfl.json.

    uv run python -m possession.bands
"""

import json
import os

from . import harness
from .params import Params

PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "bands_dfl.json")
METRICS = ("spells_per_10min", "mean_spell_s", "home_share", "d_share")
MARGIN = 0.25    # relative margin around min-max; reasoning


def compute(rows, margin=MARGIN):
    out = {}
    for k in METRICS:
        vals = [r[k] for r in rows]
        lo, hi = min(vals), max(vals)
        out[k] = [lo * (1 - margin), hi * (1 + margin)]
    return out


def load():
    if not os.path.exists(PATH):
        return None
    return json.load(open(PATH))["bands"]


if __name__ == "__main__":
    import sys
    from dataclasses import replace
    chosen = json.load(open(sys.argv[1])) if len(sys.argv) > 1 else {}
    variant = sys.argv[2] if len(sys.argv) > 2 else "A"
    p = replace(Params(), **chosen)
    rows = []
    for prof in ("gap10", "gap25", "gap75", "gap200"):
        rows += harness.evaluate_fast(p, harness.ALL_MATCHES, prof, variant)
    bands = compute(rows)
    json.dump({"source": "pipeline output on 7 degraded DFL matches x 4 gap profiles, min-max "
                         "+/- {:.0%}; variant {}; params {}".format(MARGIN, variant, chosen),
               "bands": bands}, open(PATH, "w"), indent=2)
    print(json.dumps(bands, indent=2))

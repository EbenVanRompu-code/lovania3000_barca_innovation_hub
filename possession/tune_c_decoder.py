"""Decoder grid for variant C on DFL (the v1 decoder values were tuned for variant A).
Per fold: train C once, then score a grid of decoder settings. Choose on the first 3 folds (the
tuning matches), confirm on the other 4. Never touches our match.

    uv run python -m possession.tune_c_decoder
"""

import csv
import itertools
from dataclasses import replace

import numpy as np

from . import classifier, harness, train_c
from .params import Params

GRID = {"switch_penalty": [0.5, 2, 4, 8], "dead_penalty": [2, 5], "min_hold": [25, 50, 75, 100]}

if __name__ == "__main__":
    base = Params()
    ids = harness.ALL_MATCHES
    names = list(GRID)
    cells = list(itertools.product(*[GRID[n] for n in names]))
    res = {c: {} for c in cells}
    for i, test in enumerate(ids):
        calib = [ids[(i + 1) % len(ids)], ids[(i + 2) % len(ids)]]
        trainset = [m for m in ids if m != test and m not in calib]
        classifier.ACTIVE = train_c.train(trainset, calib, base)
        for c in cells:
            p = replace(base, **dict(zip(names, c)))
            sc = [harness.evaluate_fast(p, [test], prof, "C")[0]["spellF1"] for prof in train_c.TEST_PROFILES]
            res[c][test] = float(np.mean(sc))
        print("fold", test, "done", flush=True)
    rows = []
    for c in cells:
        tune = np.mean([res[c][m] for m in ids[:3]]); conf = np.mean([res[c][m] for m in ids[3:]])
        rows.append({**dict(zip(names, c)), "tune3": tune, "confirm4": conf, "all7": np.mean(list(res[c].values()))})
    rows.sort(key=lambda r: -r["tune3"])
    with open("possession/out/tune_c_decoder.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    for r in rows[:10]:
        print({k: round(v, 4) if isinstance(v, float) else v for k, v in r.items()})
    cur = [r for r in rows if r["switch_penalty"] == 0.5 and r["dead_penalty"] == 2 and r["min_hold"] == 50][0]
    print("current v1 decoder:", {k: round(v, 4) if isinstance(v, float) else v for k, v in cur.items()})

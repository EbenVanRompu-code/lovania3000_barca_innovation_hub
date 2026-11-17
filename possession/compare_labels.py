"""Variant C trained on DFL's own fields vs on rule-based labels, each scored against BOTH label sets
(leave-one-match-out, all 7 matches, 6 degraded profiles).

    uv run python -m possession.compare_labels
"""

import json
import numpy as np

from . import harness, train_c
from .params import Params

if __name__ == "__main__":
    p = Params()
    out = {}
    for train_labels in ("field", "rule"):
        rows = train_c.lomo(p, tag="C-labels", train_labels=train_labels, score_sets=("field", "rule"))
        for ss in ("field", "rule"):
            sel = [r for r in rows if r["score_set"] == ss]
            out["train={} score={}".format(train_labels, ss)] = {
                "spellF1": harness.mean(sel), "macroF1": harness.mean(sel, "macroF1"),
                "n_pred": float(np.mean([r["n_pred_spells"] for r in sel])),
                "n_gt": float(np.mean([r["n_gt_spells"] for r in sel])),
                "mean_s": float(np.mean([r["mean_spell_s"] for r in sel])),
                "gt_mean_s": float(np.mean([r["gt_mean_spell_s"] for r in sel]))}
    for k, v in out.items():
        print(k, {a: round(b, 3) for a, b in v.items()}, flush=True)
    json.dump(out, open("possession/out/compare_labels.json", "w"), indent=1)

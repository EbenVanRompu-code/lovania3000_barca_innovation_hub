"""Train, calibrate and evaluate variant C with leave-one-match-out on DFL (PLAN.md step 9).

For each held-out test match: train on 4 of the other 6 (degraded copies), fit the temperature on
the remaining 2, score the test match on all four gap profiles. Nothing here touches our match.

    uv run python -m possession.train_c lomo
    uv run python -m possession.train_c final     # all 7 matches -> possession/models/variant_c.joblib
"""

import hashlib
import os
import sys
from dataclasses import replace

import joblib
import numpy as np

from . import ball as ball_mod
from . import classifier, harness
from .params import Params

TRAIN_PROFILES = ("gap25_j05", "gap75", "gap200_j2")   # degraded copies: gap length AND noise vary
TEST_PROFILES = ("gap10", "gap25", "gap75", "gap200", "gap75_j05", "gap75_j2")
STRIDE = 4                              # keep every 4th frame (neighbouring frames are near-duplicates)
MODEL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")
_feat = {}


def features(match_id, profile, params):
    key = (match_id, profile)
    if key not in _feat:
        tr = harness.get_tracking(match_id, profile)
        b = ball_mod.clean(tr, params)
        _feat[key] = classifier.compute_features(b, tr, params).astype(np.float32)
    return _feat[key]


def dataset(match_ids, params, profiles=TRAIN_PROFILES, labels=None):
    X, y = [], []
    for mid in match_ids:
        m = harness.get_match(mid, labels)
        keep = np.zeros(len(m.labels), dtype=bool)
        for a, b in m.blocks:
            keep[a:b + 1:STRIDE] = True
        keep &= m.labels >= 0
        for prof in profiles:
            X.append(features(mid, prof, params)[keep])
            y.append(m.labels[keep])
    return np.vstack(X), np.concatenate(y).astype(int)


def train(train_ids, calib_ids, params, seed=0, labels=None):
    X, y = dataset(train_ids, params, labels=labels)
    clf = classifier.fit(X, y, seed)
    Xc, yc = dataset(calib_ids, params, profiles=("gap25", "gap200"), labels=labels)
    return classifier.Model(clf, classifier.fit_temperature(clf, Xc, yc))


def lomo(decoder_params, ids=None, profiles=TEST_PROFILES, tag="C-lomo2", train_labels=None,
         score_sets=(None,)):
    """score_sets: label sets each fold is scored against (None = the training labels)."""
    ids = ids or harness.ALL_MATCHES
    rows = []
    for i, test in enumerate(ids):
        others = [m for m in ids if m != test]
        # deterministic split of the six others: calibrate on the two that follow `test` in id order
        calib = [ids[(i + 1) % len(ids)], ids[(i + 2) % len(ids)]]
        trainset = [m for m in others if m not in calib]
        classifier.ACTIVE = train(trainset, calib, decoder_params, labels=train_labels)
        for prof in profiles:
            for ss in score_sets:
                r = harness.evaluate_fast(decoder_params, [test], prof, "C",
                                          tag="{}|train={}|score={}".format(tag, train_labels, ss),
                                          write_log=True, label_set=ss)
                for x in r:
                    x["score_set"] = ss
                rows += r
        print("fold", test, "done", flush=True)
    return rows


def train_final(decoder_params, seed=0):
    ids = harness.ALL_MATCHES
    model = train(ids[:5], ids[5:], decoder_params, seed)
    os.makedirs(MODEL_DIR, exist_ok=True)
    path = os.path.join(MODEL_DIR, "variant_c.joblib")
    joblib.dump(model, path)
    digest = hashlib.sha256(open(path, "rb").read()).hexdigest()
    with open(os.path.join(MODEL_DIR, "variant_c.sha256"), "w") as f:
        f.write(digest + "\n")
    return path, digest


if __name__ == "__main__":
    import json
    chosen = json.load(open(sys.argv[2])) if len(sys.argv) > 2 else {}
    p = replace(Params(), **chosen)
    if sys.argv[1] == "lomo":
        rows = lomo(p)
        for prof in TEST_PROFILES:
            sel = [r for r in rows if r["profile"] == prof]
            print(prof, round(harness.mean(sel), 4))
        print("mean over all", round(harness.mean(rows), 4))
    elif sys.argv[1] == "final":
        print(train_final(p))

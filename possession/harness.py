"""Run a variant on DFL matches, score with the organisers' scorer, log every run (PLAN.md step 6).

    from possession import harness
    rows = harness.evaluate(Params(), harness.GRID_MATCHES, "gap25")

The score comes from evaluation/possession_metrics.py, used unchanged. The two DFL halves are
joined with a block of X frames so the scorer sees them as two separate runs, like our match.
Every call is appended to possession/out/runlog.csv (gitignored).
"""

import csv
import datetime
import os
import sys
from dataclasses import asdict

import numpy as np

from .external import dfl
from .external.degrade import ALL_PROFILES, degrade
from . import ball as ball_mod
from . import deadball, evidence, hmm
from .pipeline import decode_match

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "evaluation"))
import possession_metrics as pm  # noqa: E402

RUNLOG = os.path.join(ROOT, "possession", "out", "runlog.csv")
ALL_MATCHES = dfl.match_ids() if os.path.isdir(dfl.DATA) else []
# fixed split, decided before looking at any score: tune on the first 3, confirm on the other 4
GRID_MATCHES = ALL_MATCHES[:3]
CONFIRM_MATCHES = ALL_MATCHES[3:]
SEPARATOR = 100   # X frames between the halves in the scored arrays

_cache = {}
LABEL_SET = "field"     # labels used for TRAINING: "field" (DFL BallStatus/BallPossession) or "rule"


def get_match(match_id, label_set=None):
    """DFL match with the chosen label set: 'field' = DFL's own per-frame fields, 'rule' = the
    same fields filtered by the challenge's 'controlled possession' rule (external/rules.py)."""
    label_set = label_set or LABEL_SET
    key = ("match", match_id, label_set)
    if key not in _cache:
        if ("match", match_id, "field") not in _cache:
            _cache[("match", match_id, "field")] = dfl.load(match_id)
        m = _cache[("match", match_id, "field")]
        if label_set == "rule":
            from dataclasses import replace as _replace
            from .external import rules
            rows = rules.event_rows(rules.read_events(match_id), m.half_t0, m.blocks)
            m = _replace(m, labels=rules.rule_labels(m.labels, m.blocks, rows))
        _cache[key] = m
    return _cache[key]


def get_tracking(match_id, profile_name, seed=0):
    key = (match_id, profile_name, seed)
    if key not in _cache:
        m = get_match(match_id)
        if profile_name == "clean":
            _cache[key] = m.tracking
        else:
            _cache[key] = degrade(m.tracking, m.blocks, ALL_PROFILES[profile_name], seed)
    return _cache[key]


def _codes_to_pm(codes):
    """DFL codes 0 H 1 A 2 D -1 unknown -> scorer codes (X for unknown)."""
    out = np.where(codes < 0, pm.X, codes).astype(np.int8)
    return out


def _join_halves(arr, blocks, fill):
    pieces = []
    for i, (a, b) in enumerate(blocks):
        if i:
            pieces.append(np.full(SEPARATOR, fill, dtype=arr.dtype))
        pieces.append(arr[a:b + 1])
    return np.concatenate(pieces)


def score(match, pred_letters):
    """Scores for one match. pred_letters: (T,) of H/A/D/X from decode_match."""
    gt = _join_halves(_codes_to_pm(match.labels), match.blocks, pm.X)
    pred_codes = pm.label_codes(_join_halves(pred_letters, match.blocks, "X"))
    r = pm.evaluate(gt, pred_codes)
    graded = gt != pm.X
    return {
        "spellF1": r["spell_f1_mean"], "macroF1": r["macro_f1"],
        "n_pred_spells": r["n_pred_spells"], "n_gt_spells": r["n_gt_spells"],
        "mean_spell_s": r["mean_spell_duration_s"], "gt_mean_spell_s": r["gt_mean_spell_duration_s"],
        "home_share": r["pred_home_share"], "gt_home_share": r["gt_home_share"],
        "n_graded": int(graded.sum()),
        "spells_per_10min": r["n_pred_spells"] / (graded.sum() / 25 / 600.0),
        "d_share": float((pred_codes[graded] == pm.D).mean()),
        "gt_d_share": float((gt[graded] == pm.D).mean()),
    }


def log(rows):
    os.makedirs(os.path.dirname(RUNLOG), exist_ok=True)
    if os.path.exists(RUNLOG):
        with open(RUNLOG) as f:
            header = f.readline().strip().split(",")
        if header != list(rows[0]):      # columns changed: keep the old log, start a new one
            os.replace(RUNLOG, RUNLOG.replace(".csv", "_" + datetime.datetime.now().strftime("%H%M%S") + ".csv"))
    new = not os.path.exists(RUNLOG)
    with open(RUNLOG, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        if new:
            w.writeheader()
        w.writerows(rows)


def evaluate(params, match_ids, profile_name, variant="A", seed=0, tag="", write_log=True):
    """One score row per match. profile_name: 'clean' or a key of degrade.PROFILES."""
    rows = []
    stamp = datetime.datetime.now().isoformat(timespec="seconds")
    for mid in match_ids:
        m = get_match(mid)
        tr = get_tracking(mid, profile_name, seed)
        letters = decode_match(tr, m.blocks, params, None, variant)
        row = {"date": stamp, "tag": tag, "variant": variant, "match": mid,
               "profile": profile_name, "seed": seed}
        row.update(score(m, letters))
        row.update({"p_" + k: v for k, v in asdict(params).items()})
        rows.append(row)
    if write_log and rows:
        log(rows)
    return rows


def mean(rows, key="spellF1"):
    return float(np.mean([r[key] for r in rows]))


# -- fast path for grids ---------------------------------------------------------------------
# Evidence and dead-ball signals depend on only some parameters (s, c, v0, max_gap, thresholds).
# A grid over penalties, min_hold and D weights re-uses them and only re-runs the decoder.

_prep = {}


def _model_id(variant):
    if variant != "C":
        return None
    from . import classifier
    return id(classifier.ACTIVE)


def prepare(match_id, profile_name, params, variant="A", seed=0):
    key = (match_id, profile_name, seed, variant, params.max_gap, params.pitch_margin,
           params.stationary_speed, params.stationary_min_frames, params.players_slow_speed,
           params.players_slow_min_frames, params.s, params.c, params.v0,
           params.d_control, params.v_control, params.k_control, params.b_strength, params.b_mix,
           params.c_model, params.d_model, params.traj, params.traj_decay, params.traj_max,
           params.traj_tol, _model_id(variant))
    if key not in _prep:
        tr = get_tracking(match_id, profile_name, seed)
        b = ball_mod.clean(tr, params)
        ev, extra = evidence.VARIANTS[variant](b, tr, params)
        _prep[key] = (ev, extra, deadball.signals(b, tr, params), tr.n_frames)
    return _prep[key]


def evaluate_fast(params, match_ids, profile_name, variant="A", seed=0, tag="", write_log=False,
                  label_set=None):
    """label_set: which labels to SCORE against (default: the training label set)."""
    rows = []
    stamp = datetime.datetime.now().isoformat(timespec="seconds")
    for mid in match_ids:
        m = get_match(mid, label_set)
        ev, extra, sig, n = prepare(mid, profile_name, params, variant, seed)
        dead = (np.zeros(n) + extra) if variant in evidence.NO_RULES else deadball.combine(sig, params) + extra
        letters = np.full(n, "X", dtype="<U1")
        for a, b in m.blocks:
            letters[a:b + 1] = hmm.LETTERS[hmm.decode(ev[a:b + 1], dead[a:b + 1], params)]
        row = {"date": stamp, "tag": tag, "variant": variant, "match": mid,
               "profile": profile_name, "seed": seed}
        row.update(score(m, letters))
        row.update({"p_" + k: v for k, v in asdict(params).items()})
        rows.append(row)
    if write_log and rows:
        log(rows)
    return rows

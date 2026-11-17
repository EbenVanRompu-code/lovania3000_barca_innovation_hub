"""Our match -> submission CSV + plausibility report.

    uv run python -m possession.run --out possession/out/submission.csv

Run by the team (it reads data/). The report is for the team. Claude hears pass/fail in words only
(PLAN.md section 1). Step 3 has hard structural checks only; the DFL-derived bands arrive in step 6.
"""

import argparse
import os
import sys

import numpy as np

from . import ball as ball_mod
from . import deadball, evidence
from .params import FINAL_VARIANT, PRE_KICKOFF_D, Params
from .pipeline import decode_match

IMPOSSIBLE_REACH_M = 10.0   # a switch to a team whose nearest player is farther than this is suspicious


def runs_of(labels):
    """[(label, start, length)] of consecutive equal labels."""
    out, start = [], 0
    for i in range(1, len(labels) + 1):
        if i == len(labels) or labels[i] != labels[start]:
            out.append((labels[start], start, i - start))
            start = i
    return out


def hard_checks(labels, blocks, n_frames, forced):
    """name -> bool. These are invariants; a False is a bug, not a tuning matter."""
    if len(labels) != n_frames:
        return {"row count": False}
    inside = np.zeros(n_frames, dtype=bool)
    for first, last in blocks:
        inside[first:last + 1] = True
    checks = {
        "row count": True,
        "labels only H/A/D/X": bool(np.isin(labels, list("HADX")).all()),
        "no X inside the half blocks": bool((labels[inside] != "X").all()),
        "X outside the half blocks": bool((labels[~inside] == "X").all()),
    }
    if forced is not None:
        checks["2nd-half pre-kickoff frames all D"] = bool((labels[forced[0]:forced[1] + 1] == "D").all())
    return checks


def statistics(labels, tracking, ball, blocks, params, forced):
    """Numbers for the team to read. Not pass/fail; bands from DFL come in step 6."""
    fps = tracking.fps
    inside = np.zeros(len(labels), dtype=bool)
    for first, last in blocks:
        inside[first:last + 1] = True
    lab = labels[inside]
    stats = {"graded frames": int(inside.sum())}
    for c in "HAD":
        stats["share " + c] = float((lab == c).mean())
    seen = ball.observed[inside]
    stats["ball observed share"] = float(seen.mean())
    stats["ball interpolated share"] = float(ball.filled[inside].mean())
    stats["ball missing beyond max_gap share"] = float((~seen & ~ball.filled[inside]).mean())

    durations, switches, far = [], 0, 0
    d_home = evidence.nearest_distance(ball.xy, tracking.players_xy, tracking.team, 0)
    d_away = evidence.nearest_distance(ball.xy, tracking.players_xy, tracking.team, 1)
    for first, last in blocks:
        block = labels[first:last + 1]
        prev = None   # previous H/A label, ignoring D
        for lab_, start, length in runs_of(block):
            if lab_ == "D":
                continue
            durations.append(length / fps)
            if prev is not None and lab_ != prev:
                switches += 1
                dist = d_home[first + start] if lab_ == "H" else d_away[first + start]
                if np.isfinite(dist) and dist > IMPOSSIBLE_REACH_M:
                    far += 1
            prev = lab_
    d = np.array(durations)
    stats["H/A runs (possessions, a D between same-team runs splits them)"] = int(len(d))
    if len(d):
        stats["mean run s"] = float(d.mean())
        stats["median run s"] = float(np.median(d))
    stats["H<->A switches"] = switches
    stats["switches to a team >{:.0f} m from the ball".format(IMPOSSIBLE_REACH_M)] = far

    sig = deadball.signals(ball, tracking, params)
    for name, mask in sig.items():
        stats["dead signal {} frames".format(name)] = int(mask[inside].sum())
    dead = labels == "D"
    if dead[inside].any():
        stats["D frames backed by out/stationary"] = float(
            ((sig["out"] | sig["stationary"]) & dead)[inside].sum() / dead[inside].sum())
    return stats


def load_model():
    """Load the variant C model produced by `python -m possession.train_c final`, after checking
    that its hash matches the recorded one."""
    import hashlib

    import joblib

    from . import classifier
    from .train_c import MODEL_DIR
    path = os.path.join(MODEL_DIR, "variant_c.joblib")
    want = open(os.path.join(MODEL_DIR, "variant_c.sha256")).read().strip()
    got = hashlib.sha256(open(path, "rb").read()).hexdigest()
    if got != want:
        raise RuntimeError("variant_c.joblib does not match variant_c.sha256")
    classifier.ACTIVE = joblib.load(path)


def band_checks(labels, blocks, bands, fps=25):
    """name -> (value, lo, hi, within). Bands come from DFL (bands.py). A miss is a flag to
    investigate, not something to tune toward."""
    durations = []
    for first, last in blocks:
        for lab_, _, length in runs_of(labels[first:last + 1]):
            if lab_ in "HA":
                durations.append(length / fps)
    inside = np.zeros(len(labels), dtype=bool)
    for first, last in blocks:
        inside[first:last + 1] = True
    lab = labels[inside]
    alive = ((lab == "H") | (lab == "A")).sum()
    values = {
        "spells_per_10min": len(durations) / (inside.sum() / fps / 600.0),
        "mean_spell_s": float(np.mean(durations)) if durations else 0.0,
        "home_share": float((lab == "H").sum() / alive) if alive else 0.0,
        "d_share": float((lab == "D").mean()),
    }
    return {k: (v, bands[k][0], bands[k][1], bands[k][0] <= v <= bands[k][1])
            for k, v in values.items()}


def write_csv(path, labels):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", newline="\n") as f:
        f.write("frame,label\n")
        f.writelines("{},{}\n".format(i, lab) for i, lab in enumerate(labels))


def format_report(checks, stats, band_lines=None):
    lines = ["== hard checks (these should all pass) =="]
    lines += ["{:<45s} {}".format(k, "PASS" if v else "FAIL") for k, v in checks.items()]
    lines += ["", "== numbers for the team (no bands yet; bands come from DFL in step 6) =="]
    for k, v in stats.items():
        lines.append("{:<68s} {}".format(k, "{:.4f}".format(v) if isinstance(v, float) else v))
    if band_lines:
        lines += ["", "== DFL plausibility bands (a FLAG means investigate, never tune toward) =="]
        for k, (v, lo, hi, ok) in band_lines.items():
            lines.append("{:<22s} {:>9.3f}   band [{:.3f}, {:.3f}]   {}".format(
                k, v, lo, hi, "ok" if ok else "FLAG"))
    lines += ["", "OVERALL: " + ("PASS" if all(checks.values()) else "FAIL")]
    return "\n".join(lines)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", default="possession/out/submission.csv")
    parser.add_argument("--data-dir", default=None)
    parser.add_argument("--no-tags", action="store_true", help="ignore Smart Tagging")
    parser.add_argument("--variant", default=FINAL_VARIANT, choices=["A", "B", "C"])
    args = parser.parse_args(argv)

    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    sys.path.insert(0, root)
    from games import GAME, GAMES, half_video_blocks

    from . import atd
    params = Params()
    tracking = atd.load(GAME, args.data_dir)
    blocks = [(first, last) for _, first, last in half_video_blocks(GAME)]
    forced = PRE_KICKOFF_D

    if args.variant == "C":
        load_model()
    tag_factors = None
    if not args.no_tags:
        from games import asset_paths
        from . import tagging
        home, away = atd.team_names(GAME, args.data_dir)
        instances = tagging.load(asset_paths(GAME, args.data_dir)["smart_tagging"])
        tag_factors = tagging.factors(instances, tracking.n_frames, home, away, params)
    labels = decode_match(tracking, blocks, params, forced, args.variant, tag_factors)
    write_csv(args.out, labels)

    ball = ball_mod.clean(tracking, params)
    checks = hard_checks(labels, blocks, GAMES[GAME]["n_video_frames"], forced)
    stats = statistics(labels, tracking, ball, blocks, params, forced)
    if tag_factors is not None:
        # structure only: how much do the tags change the output? (never a score)
        plain = decode_match(tracking, blocks, params, forced, args.variant, None)
        inside = labels != "X"
        stats["frames where Smart Tagging changed the label (share of graded)"] = float(
            (labels[inside] != plain[inside]).mean())
    from . import bands as bands_mod
    bands = bands_mod.load()
    band_lines = None
    if bands:
        band_lines = band_checks(labels, blocks, bands)
    report = format_report(checks, stats, band_lines)
    report_path = os.path.splitext(args.out)[0] + "_report.txt"
    with open(report_path, "w") as f:
        f.write(report + "\n")
    print(report)
    print("\nwrote", args.out, "and", report_path)
    print("next: python evaluation/validate_submission.py", args.out)
    return 0 if all(checks.values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())

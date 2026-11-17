"""Rule-based DFL labels that follow the challenge's possession definition (PLAN.md section 5,
fallback; CHALLENGE.md "Possession rules").

CHALLENGE.md: possession changes only when the opposing team gains CONTROLLED possession; a simple
touch, a deflection that does not intercept, or an aerial duel without subsequent control does not
switch it; proposed proxies are "a player keeps the ball for a few seconds" or "a team completes
more than two consecutive passes".

Method (keeps DFL's exact timing): start from DFL's own per-frame labels (BallStatus gives D, the
BallPossession field gives H/A). Inside each alive stretch (between dead-ball periods) the first
possessor is kept (a restart has a clear possessor). Every later flip to the other team is kept only
if it is controlled, otherwise it is merged back into the previous possessor. A flip is controlled
when, around the run (+/- margin_s), the new team has
  * a strong event (restart, ball claiming, a tackle won with PossessionChange), or
  * at least min_events ball events (pass, cross, shot, other ball action), or
  * the run itself lasts at least keep_s seconds ("keeps the ball for a few seconds").
The three numbers are fixed by reasoning from the published definition. They are not tuned on our
match, its ground truth or its video.
"""

import xml.etree.ElementTree as ET
from dataclasses import dataclass
from datetime import datetime

import numpy as np

from . import dfl

RESTARTS = ("KickOff", "ThrowIn", "FreeKick", "CornerKick", "GoalKick", "Penalty")
BALL_EVENTS = ("Play", "ShotAtGoal", "OtherBallAction")


@dataclass(frozen=True)
class RuleParams:
    min_events: int = 2         # ball events by the new team that make a flip controlled
    keep_s: float = 3.0         # a run this long counts as "keeps the ball for a few seconds"
    margin_s: float = 1.0       # events within this margin of the run count (event clock is coarse)


def read_events(match_id, data_dir=dfl.DATA):
    """[(epoch_seconds, side 0 home / 1 away, strong)] control events, time-sorted."""
    info = dfl.read_match_info(match_id, data_dir)
    side = {info["home"]: 0, info["guest"]: 1}
    root = ET.parse(dfl._path("DFL_03_02", match_id, data_dir)).getroot()
    deleted = {e.get("EventId") for e in root.iter("Event") if e.find("Delete") is not None}
    out = []
    for ev in root.iter("Event"):
        if ev.get("EventId") in deleted or ev.get("EventTime") is None:
            continue
        t = datetime.fromisoformat(ev.get("EventTime")).timestamp()
        for ch in ev:
            team, strong = None, False
            if ch.tag in RESTARTS:
                team = ch.get("Team")
                if team is None:                       # KickOff names the team on the nested Play
                    play = ch.find("Play")
                    team = play.get("Team") if play is not None else None
                strong = True
            elif ch.tag == "BallClaiming":
                team, strong = ch.get("Team"), True
            elif ch.tag == "TacklingGame" and ch.get("PossessionChange") == "true":
                team, strong = ch.get("WinnerTeam"), True
            elif ch.tag in BALL_EVENTS:
                team = ch.get("Team")
            if team in side:
                out.append((t, side[team], strong))
    out.sort()
    return out


def event_rows(events, half_t0, blocks, fps=25):
    """[(row, side, strong)] mapped to rows of the concatenated timeline; events outside a half
    are dropped."""
    rows = []
    for t, s, strong in events:
        for t0, (a, b) in zip(half_t0, blocks):
            r = int(round((t - t0) * fps)) + a
            if a <= r <= b:
                rows.append((r, s, strong))
                break
    return rows


def _runs(x):
    out, start = [], 0
    for i in range(1, len(x) + 1):
        if i == len(x) or x[i] != x[start]:
            out.append((int(x[start]), start, i - start))
            start = i
    return out


def rule_labels(field_labels, blocks, ev_rows, params=None, fps=25):
    p = params or RuleParams()
    out = field_labels.copy()
    ev_rows = sorted(ev_rows)
    rows = np.array([r for r, _, _ in ev_rows]) if ev_rows else np.array([], dtype=int)
    sides = np.array([s for _, s, _ in ev_rows])
    strong = np.array([st for _, _, st in ev_rows], dtype=bool)
    m, keep = int(p.margin_s * fps), int(p.keep_s * fps)
    for a, b in blocks:
        seg = out[a:b + 1]
        holder = None
        for lab, start, length in _runs(seg.copy()):
            if lab not in (dfl.HOME, dfl.AWAY):          # dead or unknown: next run is a restart
                holder = None
                continue
            if holder is None or lab == holder:
                holder = lab
                continue
            lo, hi = np.searchsorted(rows, a + start - m), np.searchsorted(rows, a + start + length + m)
            mine = sides[lo:hi] == lab
            controlled = (strong[lo:hi] & mine).any() or (~strong[lo:hi] & mine).sum() >= p.min_events \
                or length >= keep
            if controlled:
                holder = lab
            else:
                seg[start:start + length] = holder       # a touch/deflection: keep the old possessor
    return out


def spell_stats(labels, blocks, fps=25):
    """(number of H/A spells, mean duration in s) with D splitting spells."""
    lengths = []
    for a, b in blocks:
        lengths += [n for lab, _, n in _runs(labels[a:b + 1]) if lab in (dfl.HOME, dfl.AWAY)]
    return len(lengths), (float(np.mean(lengths)) / fps if lengths else 0.0)

"""Smart Tagging -> extra evidence for the decoder (PLAN.md sections 3-4, step 7).

Smart Tagging is Metrica's automatic annotation: intervals with a code and tags. It is evidence,
not ground truth (DATA.md): it has false positives and false negatives, and `Team` is `N/A` on
every throw-in. Time base: video_frame = round(seconds * 25).

Step 1, set pieces (code SET PIECES): the tagged window is ~5 s around the restart (starter
        docs). ASSUMPTION: the first half of the window is dead ball waiting for the restart and
        the second half is play after it. So the first half gets a D score (w_tag_dead) and the
        second half carries the team hint of step 2.
Step 2, team hint: the `Team` tag adds a mild H/A evidence (w_tag_team per frame) over the phase
        (over the second half of a set-piece window). `N/A` or an unmatched name gives nothing.

Both weights are small next to the tracking signals, so tracking can overrule a false tag.
DFL has no Smart Tagging, so the weights are fixed by reasoning and documented, never tuned on
our match.
"""

import difflib
import re
import xml.etree.ElementTree as ET

import numpy as np

from .params import Params

SET_PIECE_CODE = "set pieces"


def load(path, fps=25):
    """[{code, start, end, tags}] with start/end as video frames (end exclusive)."""
    out = []
    for inst in ET.parse(path).iter("instance"):
        out.append({
            "code": (inst.findtext("code") or "").strip().lower(),
            "start": int(round(float(inst.findtext("start")) * fps)),
            "end": int(round(float(inst.findtext("end")) * fps)),
            "tags": {(l.findtext("group") or "").strip(): (l.findtext("text") or "").strip()
                     for l in inst.findall("label")},
        })
    return out


def _norm(name):
    return re.sub(r"[^a-z ]", "", name.lower()).strip()


def side_of(tag_team, home_name, away_name, cutoff=0.6):
    """+1 home, -1 away, 0 unknown. Fuzzy match of the tag against the metadata team names
    (the tag file writes 'Atletic Club', the metadata may write 'Athletic Club')."""
    t = _norm(tag_team or "")
    if not t or t in ("na", "n a"):
        return 0
    scores = [(difflib.SequenceMatcher(None, t, _norm(n)).ratio(), s)
              for n, s in ((home_name, 1), (away_name, -1))]
    best = max(scores)
    other = min(scores)
    if best[0] >= cutoff and best[0] - other[0] > 0.1:
        return best[1]
    return 0


def factors(instances, n_frames, home_name, away_name, params=None):
    """(ev_add, dead_add), both (n_frames,): added to the H/A evidence and the D score."""
    p = params or Params()
    ev = np.zeros(n_frames)
    dead = np.zeros(n_frames)
    for inst in instances:
        a, b = max(inst["start"], 0), min(inst["end"], n_frames)
        if b <= a:
            continue
        side = side_of(inst["tags"].get("Team"), home_name, away_name)
        if inst["code"] == SET_PIECE_CODE:
            mid = (a + b) // 2
            dead[a:mid] += p.w_tag_dead
            ev[mid:b] += side * p.w_tag_team
        else:
            ev[a:b] += side * p.w_tag_team
    return ev, dead

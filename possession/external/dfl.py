"""DFL/IDSSE open data -> the same `Tracking` arrays as our match, plus per-frame labels.

Files (external_data/idsse/, see LICENSES.md):
  DFL_02_01_matchinformation_*  teams, pitch size
  DFL_04_03_positions_raw_observed_*  25 fps positions of players, referees and ball, with the
      ball's own BallStatus (1 alive / 0 dead) and BallPossession (1 home / 2 guest) per frame

Coordinates are metres with the origin at the pitch centre; we shift them to 0..L, 0..W like ours.
The two halves are concatenated into one timeline; `blocks` says where each half sits.
"""

import os
import xml.etree.ElementTree as ET
from dataclasses import dataclass

import numpy as np

from ..atd import Tracking

HOME, AWAY, DEAD = 0, 1, 2
ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "external_data")
DATA = os.path.join(ROOT, "idsse")
CACHE = os.path.join(ROOT, "cache")


@dataclass
class DflMatch:
    match_id: str
    tracking: Tracking
    labels: np.ndarray          # (T,) 0 H, 1 A, 2 D, -1 unknown
    blocks: list                # [(first, last)] per half, inclusive frame rows
    ball_truth: np.ndarray      # (T, 2) the DFL ball (NaN none), before any degradation
    half_t0: list = None        # [epoch seconds of the first frame of each half]


def match_ids(data_dir=DATA):
    ids = sorted(f.split("_")[-1][:-4] for f in os.listdir(data_dir) if f.startswith("DFL_04_03"))
    return ids


def _path(prefix, match_id, data_dir):
    for f in os.listdir(data_dir):
        if f.startswith(prefix) and f.endswith(match_id + ".xml"):
            return os.path.join(data_dir, f)
    raise FileNotFoundError("{} for {} in {}".format(prefix, match_id, data_dir))


def read_match_info(match_id, data_dir=DATA):
    root = ET.parse(_path("DFL_02_01", match_id, data_dir)).getroot()
    general = root.find(".//General")
    pitch = root.find(".//Environment")
    return {
        "home": general.get("HomeTeamId"), "guest": general.get("GuestTeamId"),
        "pitch": (float(pitch.get("PitchX")), float(pitch.get("PitchY"))),
        "home_name": general.get("HomeTeamName"), "guest_name": general.get("GuestTeamName"),
    }


def _parse_positions(path):
    """{half: {"ball": {n: (x,y,s,status,poss)}, "players": {(team, person): {n: (x,y,s)}}}}"""
    halves = {}
    key = None
    for event, el in ET.iterparse(path, events=("start", "end")):
        if event == "start":
            if el.tag == "FrameSet":
                half = el.get("GameSection")
                team, person = el.get("TeamId"), el.get("PersonId")
                key = (half, team, person)
                halves.setdefault(half, {"ball": {}, "players": {}})
            continue
        if el.tag == "Frame" and key is not None:
            half, team, person = key
            n = int(el.get("N"))
            if team == "BALL":
                if "t0" not in halves[half] or n < halves[half]["t0"][0]:
                    halves[half]["t0"] = (n, el.get("T"))
                halves[half]["ball"][n] = (
                    float(el.get("X")), float(el.get("Y")), float(el.get("S")),
                    int(el.get("BallStatus", -1)), int(el.get("BallPossession", -1)))
            elif team != "referee":
                halves[half]["players"].setdefault((team, person), {})[n] = (
                    float(el.get("X")), float(el.get("Y")), float(el.get("S")))
            el.clear()
        elif el.tag == "FrameSet":
            key = None
            el.clear()
    return halves


def load(match_id, data_dir=DATA, use_cache=True):
    cache = os.path.join(CACHE, match_id + ".npz")
    if use_cache and os.path.exists(cache):
        z = np.load(cache, allow_pickle=False)
        tr = Tracking(z["ball_xy"], z["ball_speed"], z["players_xy"], z["players_speed"],
                      z["team"], 25)
        blocks = [tuple(b) for b in z["blocks"]]
        if "half_t0" in z.files:
            return DflMatch(match_id, tr, z["labels"], blocks, z["ball_xy"].copy(),
                            list(z["half_t0"]))

    info = read_match_info(match_id, data_dir)
    halves = _parse_positions(_path("DFL_04_03", match_id, data_dir))
    length, width = info["pitch"]
    shift = np.array([length / 2, width / 2])

    # one slot per (team, person) over the whole match
    people = sorted({k for h in halves.values() for k in h["players"]})
    slot = {k: i for i, k in enumerate(people)}
    team = np.array([HOME if t == info["home"] else AWAY for t, _ in people], dtype=np.int8)

    from datetime import datetime
    half_t0 = []
    parts, blocks, offset = [], [], 0
    for half in ("firstHalf", "secondHalf"):
        h = halves[half]
        half_t0.append(datetime.fromisoformat(h["t0"][1]).timestamp())
        ns = np.array(sorted(h["ball"]))
        n0, t = ns.min(), ns.max() - ns.min() + 1
        ball_xy = np.full((t, 2), np.nan)
        ball_s = np.full(t, np.nan)
        status = np.full(t, -1, dtype=np.int8)
        poss = np.full(t, -1, dtype=np.int8)
        for n, (x, y, s, st, po) in h["ball"].items():
            i = n - n0
            ball_xy[i] = (x, y)
            ball_s[i], status[i], poss[i] = s, st, po
        pxy = np.full((t, len(people), 2), np.nan)
        ps = np.full((t, len(people)), np.nan)
        for k, frames in h["players"].items():
            for n, (x, y, s) in frames.items():
                i = n - n0
                if 0 <= i < t:
                    pxy[i, slot[k]] = (x, y)
                    ps[i, slot[k]] = s
        lab = np.full(t, -1, dtype=np.int8)
        lab[(status == 1) & (poss == 1)] = HOME
        lab[(status == 1) & (poss == 2)] = AWAY
        lab[status == 0] = DEAD
        parts.append((ball_xy + shift, ball_s, pxy + shift, ps, lab))
        blocks.append((offset, offset + t - 1))
        offset += t

    tr = Tracking(
        ball_xy=np.concatenate([p[0] for p in parts]),
        ball_speed=np.concatenate([p[1] for p in parts]),
        players_xy=np.concatenate([p[2] for p in parts]),
        players_speed=np.concatenate([p[3] for p in parts]),
        team=team, fps=25)
    labels = np.concatenate([p[4] for p in parts])
    os.makedirs(CACHE, exist_ok=True)
    np.savez_compressed(cache, ball_xy=tr.ball_xy, ball_speed=tr.ball_speed,
                        players_xy=tr.players_xy, players_speed=tr.players_speed,
                        team=tr.team, labels=labels, blocks=np.array(blocks),
                        half_t0=np.array(half_t0))
    return DflMatch(match_id, tr, labels, blocks, tr.ball_xy.copy(), half_t0)

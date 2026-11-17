"""Our tracking (Metrica ATD, read with kloppy) -> plain arrays in metres, one row per video frame.

The array contract (`Tracking`) is the same one the DFL loader will produce in step 4, so the
rest of the pipeline never knows which source it is running on.

Slots are tracklets, not people (DATA.md): 46 slots for 22 players, and a slot can change person.
Only the team of a slot is used, never its identity.
"""

import os
import re
import sys
from dataclasses import dataclass

import numpy as np

from .params import FPS, PITCH

HOME, AWAY = 0, 1


@dataclass
class Tracking:
    ball_xy: np.ndarray        # (T, 2) metres, NaN where the ball is not seen
    ball_speed: np.ndarray     # (T,) m/s as provided by the source, NaN where not provided
    players_xy: np.ndarray     # (T, N, 2) metres, NaN where the slot is empty
    players_speed: np.ndarray  # (T, N) m/s as provided, NaN where not provided
    team: np.ndarray           # (N,) 0 = home, 1 = away
    fps: int = FPS

    @property
    def n_frames(self):
        return self.ball_xy.shape[0]


def from_frame(df, team_of_slot, n_frames, pitch=PITCH, fps=FPS):
    """Build a Tracking from a kloppy-style wide DataFrame.

    df: columns `frame_id` (counts from 1), `ball_x`, `ball_y`, optionally `ball_s` or
        `ball_speed`, and per slot `<slot>_x`, `<slot>_y`, optionally `<slot>_s`.
        Coordinates are normalised 0-1.
    team_of_slot: {slot name: 0 home / 1 away}.
    Row t of the result is video frame t (= frame_id - 1). Frames absent from df stay NaN.
    """
    length, width = pitch
    frame = df["frame_id"].to_numpy().astype(np.int64) - 1
    if frame.min() < 0 or frame.max() >= n_frames:
        raise ValueError("frame_id outside 1..{}".format(n_frames))

    ball_xy = np.full((n_frames, 2), np.nan)
    ball_xy[frame, 0] = df["ball_x"].to_numpy(dtype=float)
    ball_xy[frame, 1] = df["ball_y"].to_numpy(dtype=float)
    ball_speed = np.full(n_frames, np.nan)
    for name in ("ball_s", "ball_speed"):
        if name in df.columns:
            ball_speed[frame] = df[name].to_numpy(dtype=float)
            break

    slots = [s for s in team_of_slot]
    players_xy = np.full((n_frames, len(slots), 2), np.nan)
    players_speed = np.full((n_frames, len(slots)), np.nan)
    for j, slot in enumerate(slots):
        if "{}_x".format(slot) in df.columns:
            players_xy[frame, j, 0] = df["{}_x".format(slot)].to_numpy(dtype=float)
            players_xy[frame, j, 1] = df["{}_y".format(slot)].to_numpy(dtype=float)
        if "{}_s".format(slot) in df.columns:
            players_speed[frame, j] = df["{}_s".format(slot)].to_numpy(dtype=float)

    # Guard against a source that is not normalised 0-1 (would silently wreck every distance).
    for name, arr in (("ball", ball_xy), ("players", players_xy)):
        finite = arr[np.isfinite(arr)]
        if finite.size and (finite.min() < -0.5 or finite.max() > 1.5):
            raise ValueError("{} coordinates are not normalised 0-1".format(name))

    scale = np.array([length, width])
    return Tracking(
        ball_xy=ball_xy * scale,
        ball_speed=ball_speed,
        players_xy=players_xy * scale,
        players_speed=players_speed,
        team=np.array([team_of_slot[s] for s in slots], dtype=np.int8),
        fps=fps,
    )


def load(game=None, data_dir=None):
    """Our match: tactical ATD via kloppy. Run by the team, never by Claude (RULES_CLARIFICATION)."""
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if root not in sys.path:
        sys.path.insert(0, root)
    from games import GAME, GAMES, require
    from kloppy import metrica

    game = game or GAME
    meta, raw = require(["atd_meta", "atd_raw"], game, data_dir)
    dataset = metrica.load_tracking_epts(meta_data=meta, raw_data=raw)

    # kloppy lists the home team first (FIFA/EPTS convention, same as starter/baseline.py).
    teams = dataset.metadata.teams
    if len(teams) != 2:
        raise ValueError("expected 2 teams, found {}".format(len(teams)))
    team_of_slot = {}
    for side, team in zip((HOME, AWAY), teams):
        for player in team.players:
            team_of_slot[str(player.player_id)] = side

    df = dataset.to_df(engine="pandas")
    # Keep only slots that actually have columns, in a stable order.
    present = [c[:-2] for c in df.columns if re.fullmatch(r".+_x", c) and c != "ball_x"]
    team_of_slot = {s: team_of_slot[s] for s in present if s in team_of_slot}
    return from_frame(df, team_of_slot, GAMES[game]["n_video_frames"], GAMES[game]["pitch"],
                      GAMES[game]["fps"])


def team_names(game=None, data_dir=None):
    """(home, away) club names from the ATD metadata, first team = home (same rule as load)."""
    import xml.etree.ElementTree as ET
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if root not in sys.path:
        sys.path.insert(0, root)
    from games import GAME, require
    meta = require(["atd_meta"], game or GAME, data_dir)[0]
    names = [t.findtext("Name") for t in ET.parse(meta).getroot().iter("Team")]
    if len(names) != 2:
        raise ValueError("expected 2 teams in the metadata")
    return names[0], names[1]

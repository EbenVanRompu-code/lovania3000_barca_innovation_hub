"""Every tunable value and where it came from (PLAN.md section 9, technical document point 5).

Rule: no value here comes from our match, its ground truth or its video. Until step 6 tunes them
on DFL/IDSSE, every "reasoning" value is a placeholder and is labelled as such in SOURCES.
"""

from dataclasses import dataclass, fields

FPS = 25                    # DATA.md
PITCH = (105.3, 68.0)       # games.py, metres (length, width)

# Which evidence variant the final run uses. Set by the selection rule on DFL (PLAN.md section 6).
FINAL_VARIANT = "C"

# Frames that are graded but always D: 2nd half, before kickoff. FAQ.md: "graded from frame
# 110,808 ... does not kick off until 113,742 ... all of them are D".
PRE_KICKOFF_D = (110808, 113741)


@dataclass(frozen=True)
class Params:
    # --- speeds (ball.py) ---
    # Speeds are ALWAYS derived from positions, never read from the source: the source speed
    # columns differ between DFL and our ATD (the ATD ball speed is empty). Central difference over
    # (2 * speed_half_window + 1) frames, i.e. +/-2 frames = 0.16 s, to tame position jitter.
    speed_half_window: int = 2

    # --- ball cleaning (ball.py) ---
    max_gap: int = 25               # frames of ball gap that are linearly interpolated
    pitch_margin: float = 0.5       # m outside the lines before the ball counts as out

    # --- optional ball trajectory reconstruction (ball.py, PLAN.md section 7) ---
    traj: bool = False              # off unless it beats the noise band on held-out DFL
    traj_decay: float = 0.97        # per-frame speed decay of a rolling ball (friction)
    traj_max: int = 75              # frames (3 s) of a long gap the prediction is trusted for
    traj_tol: float = 10.0          # m, discard if the prediction misses the next sighting by more

    # --- dead-ball signals (deadball.py) ---
    stationary_speed: float = 0.3   # m/s, ball slower than this counts as stationary
    stationary_min_frames: int = 50     # 2 s of continuous stationary ball before it counts
    players_slow_speed: float = 1.2     # m/s, mean speed of tracked players below this = static
    players_slow_min_frames: int = 50   # 2 s of continuous slow players before it counts
    w_out: float = 6.0              # D score per frame: ball out of bounds (strong)
    w_stationary: float = 4.0       # D score per frame: ball stationary (strong)
    w_players: float = 2.0          # D score per frame: players slow and static (signal, not fact)

    # --- variant A evidence (evidence.py) ---
    s: float = 2.0                  # m of (d_away - d_home) that equals one unit of evidence
    c: float = 3.0                  # cap on |evidence| per frame
    v0: float = 10.0                # m/s, ball speed scale of the weight exp(-v / v0)

    # --- variant B: control events (evidence.py) ---
    d_control: float = 2.0          # m, a player this close to the ball can control it
    v_control: float = 5.0          # m/s, ball slower than this can be controlled
    k_control: int = 5              # frames in a row (0.2 s) that make a control event
    b_strength: float = 1.5         # |evidence| per frame for the holder of the last control event
    b_mix: float = 0.0              # weight of variant A's evidence added on top

    # --- variant C: how the model's probabilities enter the decoder (classifier.py) ---
    c_model: float = 4.0            # cap on |log pH - log pA| per frame
    d_model: float = 1.0            # scale of the model's D score

    # --- Smart Tagging (tagging.py) ---
    w_tag_dead: float = 1.5         # D score per frame in the first half of a set-piece window
    w_tag_team: float = 0.5         # |evidence| per frame from a Team tag (mild hint)

    # --- decoder (hmm.py) ---
    switch_penalty: float = 0.5     # cost of H<->A
    dead_penalty: float = 2.0        # cost of entering or leaving D
    min_hold: int = 50              # H/A runs shorter than this many frames are merged


SOURCES = {
    "speed_half_window": "reasoning, fixed: +/-2 frames, same for DFL and ATD (source speeds differ)",
    "max_gap": "team decision 2026-11-07 (25 frames, ~1 s)",
    "pitch_margin": "team decision 2026-11-07 (0.5 m)",
    "traj": "off by default; included only if it beats the 0.04 noise band on held-out DFL",
    "traj_decay": "reasoning, untuned: speed halves in ~1 s",
    "traj_max": "reasoning, untuned: 3 s",
    "traj_tol": "reasoning, untuned",
    "stationary_speed": "team decision 2026-11-07 (0.3 m/s); retune on DFL in step 6",
    "stationary_min_frames": "reasoning, untuned: 2 s",
    "players_slow_speed": "reasoning, untuned: walking pace is ~1.5 m/s; retune on DFL",
    "players_slow_min_frames": "reasoning, untuned: 2 s",
    "w_out": "reasoning, untuned: ball out of play is near-certain dead ball",
    "w_stationary": "reasoning, untuned",
    "w_players": "reasoning, untuned: a signal, never a hard rule; retune on DFL",
    "s": "reasoning, untuned",
    "c": "reasoning, untuned",
    "v0": "reasoning, untuned: a pass is ~10-20 m/s",
    "d_control": "reasoning, untuned: a player within ~2 m can play the ball",
    "v_control": "reasoning, untuned: dribbling is below ~5 m/s, passes are above",
    "k_control": "reasoning, untuned: 0.2 s",
    "b_strength": "reasoning, untuned: equals the largest per-state score of variant A",
    "b_mix": "reasoning, untuned; 0 = pure control events",
    "c_model": "reasoning, untuned",
    "d_model": "reasoning, untuned",
    "w_tag_dead": "reasoning, fixed (DFL has no tags); smaller than w_out so tracking can overrule",
    "w_tag_team": "reasoning, fixed (DFL has no tags); mild, below the cap c",
    "switch_penalty": "tuned on DFL (3 matches x 4 gap profiles, flat region 0.25-1), centre chosen; stage1b",
    "dead_penalty": "tuned on DFL, flat region 0.5-5, centre chosen; stage1b",
    "min_hold": "tuned on DFL: 50 frames (2 s) best of {1,5,12,25,50,75,100,150,250}; stage1/1b",
}

assert set(SOURCES) == {f.name for f in fields(Params)}, "every parameter needs a source"

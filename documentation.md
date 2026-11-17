# Documentation: what has been built so far

Covers PLAN.md steps 1–3. Written in semi-technical language so it can later feed
`docs/TECHNICAL.md`. Nothing here has been run on the real match yet. Everything was tested on a
scripted fake match (see "How it was tested").

**Goal in one sentence:** for every video frame of the Barcelona–Athletic match, say who has the ball
(`H` home, `A` away), whether the ball is dead (`D`), or that the frame is not graded (`X`).

---

## Overview: how the pieces fit

```
tracking file ──► atd.py ──► ball.py ──► evidence.py ──┐
 (players+ball)    arrays     clean ball   who is closer │
                                │                       ├──► hmm.py ──► labels ──► run.py ──► CSV + report
                                └──► deadball.py ───────┘    (decoder)
```

1. **Load** the tracking into plain arrays in metres (`atd.py`).
2. **Clean the ball**: fill short gaps, compute speed, spot "outside the pitch" (`ball.py`).
3. **Score each frame** two ways: which team looks like it has the ball (`evidence.py`), and how
   "dead" the frame looks (`deadball.py`).
4. **Decode** the whole half at once so the labels are stable and don't flicker (`hmm.py`).
5. **Write** the CSV and a self-check report (`run.py`).

All numeric settings live in one place, `possession/params.py`, each with a note on where the value
came from.

---

## Step 1: Loading the tracking and cleaning the ball

### `possession/atd.py`: tracking → arrays
- Reads the automatic tracking data (ATD) with `kloppy`, as decided.
- Produces a `Tracking` object with:
  - `ball_xy`: ball position per frame, in metres (NaN = not seen)
  - `players_xy`: positions of all 46 "track slots", in metres (NaN = empty slot)
  - speeds as given by the source (NaN where not given)
  - `team`: for each slot, 0 = home, 1 = away
- **Metres, not fractions:** the source stores positions as 0–1 fractions of the pitch. We multiply by
  the pitch size, 105.3 × 68.0 m, so distances mean something.
- **Frame numbering:** the source counts frames from 1, the submission from 0, so
  `video_frame = frame_id − 1`. Frames missing from the source stay NaN.
- **Slots are not people.** There are 46 slots for 22 players, and a slot can switch person. We use
  only which team a slot belongs to, never who it is.
- **Safety check:** if coordinates are not in the 0–1 range, loading stops with an error. Otherwise
  every distance would be silently wrong.
- **Not yet verified on real data:** the name of the ball-speed column and how slots map to teams are
  assumptions. If they are wrong the loader will raise an error.

### `possession/ball.py`: the ball is missing ~41% of the time
- **Short gaps (up to 25 frames, about 1 second):** the ball position is filled in as a straight
  line between the last and next sighting, but only if the ball is seen on both sides of the gap.
  Nothing is extrapolated past the end of the data.
- **Longer gaps:** left empty. The decoder then keeps its previous decision through the gap.
- **A missing ball is never read as "dead ball".**
- **Ball speed:** the source's own speed where given, otherwise distance moved since the previous
  frame × 25.
- **Out of pitch:** the ball counts as out when it is more than 0.5 m beyond a touchline or goal
  line.

---

## Step 2: Evidence, dead ball, and the decoder

### `possession/evidence.py`: variant A, "who is closer to the ball"
For each frame:

```
evidence = clip( (d_away − d_home) / s , −c , +c ) × exp( −ball_speed / v0 )
```

| symbol | meaning | value |
|---|---|---|
| `d_home`, `d_away` | distance (m) from the ball to the nearest home / away player | measured |
| `s` | metres of difference that count as one unit of evidence | 2.0 |
| `c` | cap, so one frame can't dominate | 3.0 |
| `v0` | ball speed (m/s) at which the evidence weight falls to about 37% | 10.0 |

Positive evidence means home, negative means away. A fast ball (a pass in flight) gives weak
evidence because speed says little about who controls it. With no ball position, or no tracked
player on one team, evidence is 0: the frame says nothing.

### `possession/deadball.py`: is the ball dead?
Four rules. Rules 1–3 are **signals** that raise the score for `D`; none of them forces a label.
Only rule 4 is a fixed fact.

1. **Ball outside the pitch.** Strong signal.
2. **Ball stationary:** speed under 0.3 m/s for at least 2 s in a row. Strong signal.
3. **Players slow and static:** the average speed of tracked players under 1.2 m/s for at least 2 s
   in a row. Weaker signal. It works even when the ball is missing. The team considers it the most
   applicable rule, so it is the one to tune carefully on DFL data.
4. **Fixed:** 2nd-half frames 110,808–113,741 are always `D` (published in FAQ.md).

Weights are 6 (out), 4 (stationary) and 2 (players slow) per frame. They are untuned placeholders.

### `possession/hmm.py`: the decoder
Looking at each frame alone would produce noisy flicker, so the whole half is decided at once with a
**Viterbi decoder** (the standard method for finding the best sequence of states).

- **Three states:** `H`, `A`, `D`.
- **Per-frame score for each state:** `H` gets +evidence/2, `A` gets −evidence/2, `D` gets the
  dead-ball score. A frame with no information scores 0 for all three, so the state carries on.
- **Switching costs a penalty:**
  - `H ↔ A`: 20
  - entering or leaving `D`: 10
  - staying is free.
  
  The decoder only switches when enough evidence has built up to pay the penalty. These numbers are
  on an arbitrary scale tied to `s` and `c`; they are placeholders for tuning on DFL.
- **Post-filter, `min_hold = 25` frames (1 s):** an `H` or `A` run shorter than 1 s (a deflection or
  one touch) is merged into the longer neighbouring run. `D` runs are never removed.

### `possession/pipeline.py`: putting it together
`decode_match(tracking, blocks, params, forced_dead)` returns one label per video frame. Inside the
two half blocks the output is only `H`, `A` or `D`, **never `X`**. Outside them it is `X`.

---

## Step 3: Running it on our match

### `possession/run.py`
```
uv run python -m possession.run --out possession/out/submission.csv
```
- Loads our tracking, uses the half blocks from `games.py`, applies the fixed `D` block, decodes and
  writes `frame,label` rows for all 199,319 frames.
- Also writes `<out>_report.txt` next to the CSV and prints it.
- **Run by the team only.** It reads `data/`, which Claude never touches. The team reports back in
  words, never numbers.

### The report
**Hard checks (must all say PASS):** row count is right; only `H/A/D/X` appear; no `X` inside the
halves; `X` everywhere outside them; the 2nd-half pre-kickoff frames are all `D`.

**Numbers for the team to read** (they are not pass/fail yet because the acceptable ranges will come
from DFL data in step 6):
- share of `H`, `A` and `D`
- number of possessions, mean and median length in seconds
- how much of the ball was seen, interpolated, or missing beyond 1 s
- how many frames each dead-ball signal flagged
- the number of `H↔A` switches, and how many went to a team whose nearest player was more than
  10 m from the ball (a rough "impossible switch" check)
- the share of `D` frames backed by "ball out" or "ball stationary"

### After running
`python evaluation/validate_submission.py possession/out/submission.csv` checks the file format only.
No scoring against any ground truth happens until the very end of the project.

---

## How it was tested

`tests/synth.py` builds a **scripted fake match**: 46 slots (22 used), a ball that moves between
teams, a short and a long ball gap, a stretch where the ball is out, and a stretch where it sits
still. The true labels are known, so tests can check the answers exactly. There are 21 tests (loader,
ball cleaning, evidence, dead ball, decoder, post-filter, pipeline, report, CSV format).

On the fake match the decoder matched the script about 99.5% of the time; the few misses are the
frames right at a change. **This shows the code is wired correctly. It does not measure real
accuracy**: real tracking is much noisier. Real accuracy is judged on DFL data (step 5 onward).

Run the tests with `.venv/bin/python -m pytest`.

---

## Parameters at a glance (`possession/params.py`)

| name | value | source |
|---|---|---|
| `max_gap` | 25 frames | team decision |
| `pitch_margin` | 0.5 m | team decision |
| `stationary_speed` | 0.3 m/s | team decision |
| `stationary_min_frames` | 50 (2 s) | reasoning, untuned |
| `players_slow_speed` | 1.2 m/s | reasoning, untuned |
| `players_slow_min_frames` | 50 (2 s) | reasoning, untuned |
| `w_out`, `w_stationary`, `w_players` | 6, 4, 2 | reasoning, untuned |
| `s`, `c`, `v0` | 2.0, 3.0, 10.0 | reasoning, untuned |
| `switch_penalty`, `dead_penalty` | 20, 10 | reasoning, untuned |
| `min_hold` | 25 frames | reasoning, untuned |

"Untuned" means a placeholder from football reasoning. Step 6 re-fits these on DFL/IDSSE data. None
may ever be set from our match, its ground truth, or its video.

---

## Known limitations so far
- Never run on real data. The loader's kloppy column names are assumptions.
- Penalties and weights are placeholders, so the first submission is a safety net and not a tuned
  result.
- A stationary-ball threshold of 0.3 m/s may behave differently on noisy camera-based tracking than
  on cleaner data. The report shows whether rule 2 fires at all; any retuning is done on DFL.
- The ball height is unknown, so a ball flying over a player still looks like a nearby player.
- Smart Tagging is not used yet (step 7).

## Files added or changed
- New: `possession/` (`params.py`, `atd.py`, `ball.py`, `evidence.py`, `deadball.py`, `hmm.py`,
  `pipeline.py`, `run.py`), `tests/` (`synth.py`, `test_step1.py`, `test_step2.py`,
  `test_step3.py`), `pytest.ini`, `documentation.md`.
- Changed: `pyproject.toml` (a `dev` group with `pytest`) and `uv.lock`.
- Nothing is committed.

---

# Steps 4–10 (added in the second build session)

> Everything in this part was built and run **only on external DFL/IDSSE data and on synthetic
> data**. Nothing here read `data/`, the public ground truth, or any output of our match.

## Step 4: DFL/IDSSE open data: licence, download, loader, labels

**Licence.** Read from the Figshare API record on 2026-11-07: **CC BY 4.0**
(https://creativecommons.org/licenses/by/4.0/). The same statement in the Scientific Data paper is
*not yet verified*. Someone on the team should read it and paste the sentence into
`possession/external/LICENSES.md`. Attribution text for the technical document is already there.

**Download** (`possession/external/download.py`). Pulls all 21 files (7 matches × match information,
events, positions) into `external_data/idsse/` (gitignored, about 2.6 GB). Each file is checked
against the md5 Figshare publishes. All 21 matched.

**What the data contains** (confirmed by reading the files):
- 25 fps positions for players, referees and the ball, in metres with the origin at the pitch
  centre (we shift it to 0…105 × 0…68 like our data);
- per frame, the **ball's own `BallStatus`** (1 alive, 0 dead) and **`BallPossession`** (1 home,
  2 guest). These are what we use as labels: `D` where the ball is dead, `H`/`A` otherwise;
- the DFL ball is never missing, unlike ours.

**Loader** (`possession/external/dfl.py`). Streams the large XML files and builds the *same
`Tracking` arrays* as our match loader, so the pipeline runs unchanged on both sources. The two
halves are joined into one timeline and `blocks` records where each half is. A parsed match is
cached as a compressed `.npz` in `external_data/cache/` (also gitignored), so the 20-second parse
runs once.

**Label sanity check.** When the label says home has the ball, a home player is the nearest to the
ball 76% of the time; when it says away, only 20% of the time. So home/away are not swapped.

**Important difference to note.** In DFL, the ball is dead about 41% of the time and its possession
flag changes often (a mean spell of roughly 5–7 s). The organisers' definition ("deflections and
single touches do not switch") may produce longer possessions. We cannot tune toward the
organisers' labels (rules), so this is a known mismatch (see Limitations).

## Step 5: Degrading DFL to look like our tracking, and the scoring harness

**Degradation** (`possession/external/degrade.py`). Only *documented* facts about our data are
used. We do not measure anything on our match.
- The ball is hidden on **41%** of frames (DATA.md), as alternating seen/missing runs with random
  (geometric) lengths. We do **not** know our real gap lengths, so we sweep four profiles with a
  mean gap of **10, 25, 75 and 200 frames** (`gap10 … gap200`) and prefer settings that work for all
  of them.
- Small position noise on ball (0.2 m) and players (0.1 m), and short player dropouts (about 4% of
  slot-frames). These magnitudes are reasoning, not measurements.
- Fully deterministic given a seed.

**Harness** (`possession/harness.py`). Runs any variant on any DFL match, clean or degraded, and
scores it with the organisers' `evaluation/possession_metrics.py`, **unchanged**: spellF1,
macroF1, possession count, mean duration, home share and D share. The two halves are separated by
a block of `X` frames so the scorer treats them as two runs, as it does for our match. Every run
can be appended to `possession/out/runlog.csv` (variant, all parameters, match, profile, scores,
time). A fast path re-uses the evidence and signals when only decoder settings change.

**Fixed split, decided before looking at any score:** tune on the first three matches
(`DFL-MAT-J03WMX`, `…WN1`, `…WOH`), confirm on the other four.

## Step 6: Tuning variant A on DFL

Protocol (fixed in advance, in `possession/tune.py`): objective is the mean spellF1 over the 3 tuning
matches and the 4 gap profiles; the worst profile is also reported; prefer the centre of a flat
region; within 0.04 spellF1 (the scorer's noise band) take the simpler option.

### Step 6 results (variant A)

| what | result (mean spellF1 over the 4 gap profiles) |
|---|---|
| placeholders from step 3 | 0.16 on the 3 tuning matches, 0.20 on the 4 confirmation matches |
| **tuned decoder** (`switch_penalty` 0.5, `dead_penalty` 2, `min_hold` 50 frames = 2 s) | **0.30** / **0.31** |

- The score was flat across a broad area (switch penalty 0.25–1, dead penalty 0.5–5); I took the
  centre of it. `min_hold` = 50 beat 1, 5, 12, 25, 75, 100, 150 and 250. In practice the 2-second
  merge does most of the smoothing; the Viterbi penalties barely matter at these values.
- The confirmation matches (not used for tuning) scored as well as the tuning matches, so the
  settings are not overfitted.
- **Stage 2** (one parameter at a time: `s`, `c`, `v0`, the dead-ball weights and thresholds):
  nothing moved the score by more than ~0.01 except `v0 = 5` and `s = 8`, which were worse. The
  defaults stay (inside the noise band, so the simpler option wins). The hand-written dead-ball
  signals have no measurable effect on DFL, whose ball is clean; their value is untested on noisy
  data.
- Pitfall found: because `H→D→A` costs two dead penalties, the real cost of a team switch is
  `min(switch_penalty, 2 × dead_penalty)`.
- Run log: `possession/out/runlog.csv` (every run: variant, all parameters, match, profile, scores).

**Plausibility bands** (`possession/bands.py` → `bands_dfl.json`): computed from the *held-out* output
of the chosen variant on the 7 degraded DFL matches, min–max ± 25%. They are used by `run.py` to
flag (never to tune): possessions per 10 min (18.6–41.1), mean possession length (6.2–17.2 s), home
share (0.31–0.86) and D share (0.30–0.68).

## Step 7: Smart Tagging (`possession/tagging.py`)

Reads the tagging XML, converts times to frames (`round(seconds × 25)`) and returns two extra score
arrays that the decoder adds to its evidence:
- **Set pieces** (throw-in, corner, goal kick, free kick, kick-off): a window of about 5 s around the
  restart. *Assumption:* the first half of the window is dead ball and the second half is play, so
  the first half gets a weak `D` score (1.5 per frame) and the second half carries the team hint.
- **Team tag:** a mild hint (0.5 per frame) for home or away over the phase. `N/A` (every throw-in)
  or an unmatched name gives nothing. Names are matched to the ATD metadata by fuzzy text match
  ("Atletic Club" vs "Athletic Club").
- Both weights are small next to the tracking signals, so **tracking can overrule a false tag**
  (tested). The weights are fixed by reasoning and never tuned, because DFL has no tags.
- `run.py` reports only *how much the tags changed the output* (share of frames), never a score.
  `--no-tags` switches them off.
- Caveat: a long stretch with no ball evidence can be decided by the tag alone, because a mild hint
  adds up over time.

## Step 8: Variant B, control events

A control event is a player within 2 m of the ball, ball slower than 5 m/s, for 5 frames in a row.
The team of the last such event holds the ball (strong evidence, 1.5 per frame) until the opponent
gets one. Passes and deflections never switch it. **Result: spellF1 0.16 / 0.17, far worse than A**
(also with A's evidence mixed in). Rejected. Possible reasons: with a noisy ball the control events
are too sparse, and holding the last team through long gaps is too confident.

## Step 9: Variant C, learned evidence (`classifier.py`, `train_c.py`)

- Model: scikit-learn `HistGradientBoostingClassifier`, three classes, seeded, **instead of
  LightGBM**. Same model family; LightGBM would have needed a system library (`brew install
  libomp`) on this Mac. Probabilities are calibrated with one temperature fitted on held-out matches.
- 34 features, direction-free so they transfer between matches (teams swap ends at half time):
  distances of the nearest and second-nearest home and away player to the ball and their difference;
  ball speed, turning, distance to the lines, out-of-pitch flag; mean player speed and number of
  walking players; players tracked per team; team spread and centroid distance; ball-gap length and
  whether the ball is seen; and past/future averages (1 s and 3 s) of the distance difference, ball
  speed, player speed and ball visibility.
- Trained on degraded DFL copies (gap profiles 25 and 75), 1 frame in 4, calibrated on gap profiles
  25 and 200.
- **Leave-one-match-out over all 7 matches** (train on 4, calibrate on 2, test on the held-out one,
  evaluated on all four gap profiles): mean spellF1 **0.45** (0.49 / 0.47 / 0.43 / 0.40 for gap
  profiles 10 / 25 / 75 / 200); per match 0.39–0.50. macroF1 about 0.86 vs 0.64 for A; the share of
  D is close to the labels (46% vs 45%).
- It beats A by about 0.14, far above the 0.04 noise band, and wins on every match. **C is the
  chosen variant** (`FINAL_VARIANT = "C"`).
- Final model: trained on 5 matches, calibrated on 2, saved to
  `possession/models/variant_c.joblib`, with its SHA-256 in `variant_c.sha256`. `run.py` refuses
  to run if the hash does not match.
- Caveat: C learns DFL's definition of possession and dead ball (including how often it flips).
  If the organisers' definition differs, C's advantage may shrink. A and C use the same decoder.

## Step 10: Trajectory reconstruction (optional, `ball.py`, flag `traj`)

In a long gap the ball is extrapolated from its last velocity with friction (3 s at most), the
prediction is dropped if it misses the next sighting by more than 10 m, and a predicted path that
leaves the pitch counts as out. **Result: no gain (0.296 vs 0.303 and 0.311 vs 0.314), so it stays
off.** The code and tests remain behind the flag.

## Step 11: Final run and checks (to be done by the team)

Everything that touches `data/` is run by the team; tell Claude results in words only.

1. `uv sync`
2. `uv run python -m possession.run --out possession/out/submission.csv`
   - The report lists the hard checks (must say PASS), the numbers, and the DFL band checks
     (a FLAG means investigate, never tune toward).
   - If it errors: send only the exception type and message (no printed values).
3. `python evaluation/validate_submission.py possession/out/submission.csv` (format only).
4. `uv run python -m possession.check_repro`: runs twice and compares file hashes.
5. **Once, at the very end:** score the public 5-minute ground truth (`evaluation/run_evaluation.py`)
   only as validation: alignment, home/away not swapped, labels mean what we think. A bug it exposes
   may be fixed. **No parameter is changed because of it, and it never chooses between variants.**
6. Optional comparison, structure only: rerun with `--variant A` to see how different the output is.

**Not yet verified on real data (watch for these on the first run):** the kloppy column names for
ball and player speed, the slot-to-team mapping, the ATD speeds being on a similar scale to DFL's
(variant C uses them), and the fuzzy matching of the Smart Tagging team names.

## Step 12: Technical document and clean-up

- `docs/TECHNICAL.md` is drafted (approach, assets, training data and licence, rules, limitations,
  reproduction, compliance). Still to add by the team: the exact licence sentence from the paper,
  the final public-GT validation sentence, and any run-time observations.
- Files to keep out of the public repo (already gitignored): `data/`, `external_data/`,
  `possession/out/`. The model file `possession/models/variant_c.joblib` is derived from DFL only
  and is meant to be submitted. Check `git status` before any push.
- Nothing is committed.

## Final checklist (16 Nov)
- [ ] Run `possession.run`; all hard checks PASS; no unexplained FLAG
- [ ] `validate_submission.py` passes
- [ ] `check_repro` says IDENTICAL; run once more in a clean environment
- [x] Public GT used only as validation (several scorings, none used to choose a parameter or variant); stated in the technical document
- [x] Licence sentence from the paper pasted into `possession/external/LICENSES.md` (done 17 Nov)
- [ ] `docs/TECHNICAL.md` complete; files named `<team_name>.csv` and `<team_name>.md` or `.pdf`
- [ ] Public repo has no `data/`, `external_data/`, outputs or match CSV
- [ ] Google Form submitted (final version only)

## Updated files list
New: `possession/` (`params.py`, `atd.py`, `ball.py`, `evidence.py`, `deadball.py`, `hmm.py`,
`pipeline.py`, `tagging.py`, `classifier.py`, `train_c.py`, `harness.py`, `tune.py`, `bands.py`,
`bands_dfl.json`, `check_repro.py`, `run.py`, `external/` with `download.py`, `dfl.py`,
`degrade.py`, `LICENSES.md`, and `models/`), `tests/` (steps 1–10, 42 tests), `docs/TECHNICAL.md`,
`pytest.ini`, `documentation.md`. Changed: `pyproject.toml` and `uv.lock` (added `scikit-learn`
and, for tests, `pytest`).

---

# Fix after the first run on our match (17 Nov)

**What went wrong.** The first run on the real match passed every format and hard check, but the
file was more than 90% `D` and three of the four DFL plausibility bands were flagged. The public
ground-truth scorer was run once and showed the first 5 minutes of play had no `H`/`A` at all. This
was a bug, not a home/away swap (the home team is Barcelona and the slot-to-team mapping is
consistent) and not an alignment problem.

**Cause.** Variant C learned on DFL's own clean, smoothed speed columns. Our ATD has an empty ball
speed column, and its player speeds are on a different scale, so the speed features (for example
"players walking") looked like a stoppage almost everywhere.

**Fix (a train/inference mismatch, not a tuning step).**
- All speeds, for the ball and the players, are now **derived from positions**, identically for
  DFL and our ATD, and the source speed columns are ignored. Speed is a central difference over
  ±2 frames (0.16 s) to tame jitter (`speed_half_window`). The ball's turning is also measured over 5
  frames.
- C was **retrained on DFL** with training copies that vary both gap length *and* position noise
  (ball 0.1–0.4 m, players 0.05–0.2 m), because we have not measured our own noise.
- No threshold or parameter was set from our match, its ground truth, or the video.
- Leave-one-match-out over the 7 DFL matches: mean spellF1 **0.45** (0.51 / 0.48 / 0.45 / 0.41 for
  gap profiles 10 / 25 / 75 / 200; 0.43 at low noise and 0.44 at high noise), so it is robust to the
  noise level.

**Re-run on our match.** All hard checks pass, the organisers' format validator passes, **no band
is flagged any more**, two runs produce byte-identical files, and the first 5 minutes now contain
`H` and `A`. About 50–70% of graded frames are `D`, which is inside the DFL band (30–68%) but on
the high side.

**Notes for the technical document.**
- The hand-written dead-ball thresholds (variants A and B) were never used by the submitted variant C.
- The final model's hash is in `possession/models/variant_c.sha256`; `run.py` refuses to run if it
  does not match.
- The public ground truth was scored several times for validation only (first on the broken file,
  which exposed the bug; then on the fixed file). It must not be used to choose any parameter or
  variant.

---

# Improvement attempts after v1 (branch `rule-based-labels`): neither adopted

Both ran on DFL data only (leave-one-match-out, 7 matches, 6 degraded profiles). The submitted v1
file (`phase1-possession-v1`, commit `a1b9b08`) is unchanged.

## 1. Rule-based DFL labels (`possession/external/rules.py`)
Idea: the model learns DFL's own possession flag, which flips about every 6–7 s, while the
challenge says possession changes only on *controlled* possession. So the labels were filtered by
the published rule, keeping DFL's exact timing. Inside an alive stretch a flip to the other team is
kept only if it is controlled; otherwise it is merged back into the previous possessor. A flip is
controlled if the new team has a strong event (restart, ball claiming, tackle won with possession
change), or at least 2 ball events (pass, cross, shot, other ball action) within ±1 s, or the run
lasts at least 3 s. These three values come from the wording of CHALLENGE.md and were fixed before
looking at any result. About 5% of alive frames change; spells drop from about 450 to about 260 per
match and the mean length goes from about 7 s to about 12 s.

| model trained on | scored on DFL's own fields | scored on rule labels |
|---|---|---|
| DFL fields (v1) | 0.454 | 0.578 |
| rule labels | 0.435 | 0.581 |

**Not adopted.** No gain, even on the rule labels (+0.003), and slightly lower on DFL's fields
(−0.019); both inside the 0.04 noise band. v1 already predicts about 277 spells with a mean of 11 s
per match, close to the rule labels (259 spells, 12 s), because the decoder's 2-second `min_hold`
already does the same smoothing. Scores on rule labels are higher only because those labels are
smoother, so they cannot be compared across label sets. Note: the public ground truth cannot decide
between label sets (it is not allowed to choose between variants).

## 2. Decoder retuned for variant C (`possession/tune_c_decoder.py`)
The decoder values were tuned for variant A. A grid over switch penalty (0.5–8), dead penalty
(2, 5) and `min_hold` (25–100) for variant C: the best setting (4 / 5 / 25) scored 0.463 on all 7
matches against 0.454 for the current (0.5 / 2 / 50), inside the noise band. **No change.**

## Conclusion
v1 stays as the submission. The remaining weak points seen on the public 5 minutes (a few extra
switches, possessions slightly short) did not respond to either change on DFL, so they are not
fixable cheaply without a new idea. Not tried: retraining the final model on all 7 matches instead
of 5 (small expected gain, unverifiable on DFL), extra features, ensembling seeds.

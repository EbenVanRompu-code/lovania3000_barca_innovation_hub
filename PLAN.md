# Master plan: Phase 1 ball possession

Merged from both team plans on 2026-11-07. Team of two, pairing on everything. Most of the work
happens on 7 Nov, with small slots next week.
**Deadline: 17 Nov 2026.** Goal: win on `spellF1`; the baseline to beat is 0.2433.

The binding rules are in [RULES_CLARIFICATION.md](RULES_CLARIFICATION.md). Where anything conflicts,
the stricter reading applies.

---

## 1. Compliance and working rules

**Hard rules**
- **This match is input only.** Training, fitting, tuning and calibration use **external data
  only** (DFL/IDSSE), with the licence checked and credited. That includes "unsupervised"
  statistics: we don't measure gap lengths, speeds or any other statistic on our match to set a
  value.
- **No hand labels or hand corrections** on the provided match, in any amount. No human-produced
  data about it either: event data, lineups, roles, corrected tracking.
- **No checking output against the video.** Watching it to understand what counts as possession is
  allowed.
- **No match data to AI services.** That covers the data itself, extracts, derived rows, and
  numbers from runs on our match.
- **Reproducible:** the public repo regenerates the final CSV exactly. `data/` and every derived
  file stay out of it.

**Who runs what (strictest reading, no organiser email)**
- **Claude** writes the code and its tests, and runs them on synthetic data and DFL data. It never
  reads `data/`, any submission CSV for our match, or any scorer output on our match.
- **The team** runs everything that touches `data/`. Results of runs on our match go to Claude in
  **words only** ("works", "error: …", "plausibility checks pass/fail on X"), **never numbers**.
- **Public 5-minute GT:** used **only at the end**, as a final validation and plausibility check
  (alignment, H/A not swapped, labels mean what we think). A bug it exposes may be fixed. **No
  parameter is ever changed because of it**, and it is never used to choose between variants.
- **Lineups and roles:** not used.

---

## 2. Inputs

| asset | use |
|---|---|
| tactical tracking (players, ball, speeds, team per slot) | **main input** |
| Smart Tagging | **input:** set pieces as strong evidence, team tags as a mild hint |
| pitch size 105.3 × 68.0 m | from `games.py` |
| half boundaries; 2nd half frames 110,808–113,741 are `D` | published in DATA.md / FAQ.md, not read off the match |
| homography, video, broadcast feed | **not used** (state the reason in the technical document) |
| DFL/IDSSE open data (7 matches, 25 fps, tracking + events) | **external:** training, tuning, validation, plausibility bands |

---

## 3. Model: one decoder, evidence variants A → B → C

All variants share **one decoder**, so any score difference comes from the evidence, not from the
smoothing. Each variant is a valid submission. A variant replaces the previous one only if it wins
on DFL under the selection rule (§6).

### Shared decoder: Viterbi over H / A / D, per half
- **Emissions:** per-frame evidence from the active variant, plus the Smart Tagging factor (§4).
  A frame without usable ball evidence gets a **flat emission**, so the state carries through the
  gap.
- **Transitions:** staying is free. `H↔A` costs `switch_penalty`; entering or leaving `D` costs
  `dead_penalty`. Starting values come from transition counts on the DFL labels, and are then tuned
  on DFL.
- **Post-filter / hysteresis:** a run shorter than `min_hold` frames is merged into the surrounding
  possession. The search range runs from 1 frame to several seconds.
- **Output:** H/A/D on every frame inside the two half blocks. **Never X there.** X outside them.

### Variant A: geometric evidence (tonight)
- Distance of the nearest home and nearest away player to the ball, plus ball speed. The closer
  team scores higher, and a fast-moving ball scores low for both.
- Fewest parameters. This is the safety-net submission.

### Variant B: control events
- A control event is a player within `d_control` of the ball, with ball speed under `v_control`,
  for at least `k` frames.
- That team gets strong evidence until an opposing control event. Passes and deflections without
  control do not switch possession.

### Variant C: learned evidence
- **LightGBM**, three classes H/A/D, trained on DFL. **Probability calibration** on held-out matches.
- **Features:**
  - ball to nearest and second-nearest home and away player
  - ball speed, change of direction, inside/outside the pitch, distance to the nearest line
  - mean player speed, number of players walking, team centroids and spread
  - the above averaged over past and future windows (~1–3 s)
  - ball-gap length
- Trained on **degraded** DFL tracking (§5), validated **leave-one-match-out** (never split by
  frame).
- Seeded and deterministic. The model file is versioned by hash and read by the run.

---

## 4. Ball gaps, dead ball, Smart Tagging

**Ball gaps (41% of graded frames)**
- Gap up to `max_gap` frames: linear interpolation of the ball position.
- Longer gap: no ball evidence, so the emission is flat and the decoder carries the state. Variant
  C still gets the gap-length and player clues.
- A missing ball alone is never treated as dead ball.

**Dead ball**
1. Ball outside the pitch bounds (after interpolation).
2. Ball stationary beyond a threshold for a minimum time.
3. Players slow and static (stoppage behaviour).
4. Fixed: the 2nd-half frames before kickoff (110,808–113,741) are `D`, as stated in FAQ.md.

Thresholds start from football reasoning and are tuned on DFL using its ball-status field.

**Smart Tagging** (`video_frame = round(seconds * 25)`)
- **Step 1:** set pieces (throw-in, corner, goal kick, free kick, kickoff) are **strong evidence**
  for `D` shortly before the restart and mark where a new possession starts. Tracking can overrule
  a false tag.
- **Step 2:** the `Team` tag on a phase is a **mild hint** for H vs A. `N/A` (every throw-in) gives
  no team evidence.
- DFL has no Smart Tagging, so these weights are **few, fixed by reasoning and documented**. Their
  effect is judged only on structure (plausibility checks, runs with vs without), never on a score
  for our match.

---

## 5. External data: DFL / IDSSE

**Step 1: verify before relying on it**
- Licence: read the CC BY 4.0 statement on Figshare (doi 10.6084/m9.figshare.28196177) and in the
  Scientific Data paper (doi 10.1038/s41597-025-04505-y). Record the exact wording and the
  attribution in `possession/external/LICENSES.md`. Treat it as unconfirmed until then.
- Contents to confirm: 7 matches, 25 fps, ball and player tracking, events, and **per-frame ball
  status (alive/dead) and ball possession fields**.

**Step 2: storage.** `external_data/idsse/`, gitignored. Never committed, even though the licence
would allow it.

**Step 3: loader.** Use kloppy's Sportec loader if it handles the files, otherwise a small parser.
Convert to the same internal arrays as our match (metres, 25 fps, team per player), so the
pipeline runs unchanged on both sources.

**Step 4: labels**
- **Primary:** DFL's own per-frame ball-status and ball-possession fields. That's the same kind of
  source the organisers used ("corrected tracking: dead flag + owning team").
- **Fallback:** event-derived labels following the challenge rules. Possession switches only on a
  controlled action by the other team; deflections, duels and single touches don't switch; `D`
  from ball out until the restart.
- Known gap: the definitions will differ around loose balls and deflections. Use the labels to
  choose robust parameters, and expect some mismatch.

**Step 5: degrade DFL to look like our tracking**
- DFL's ball is near-complete, ours is missing 41%. Drop ball frames in **contiguous gaps** to the
  **documented** ~41% rate (DATA.md), not a rate measured by us.
- The gap-length distribution is not measured on our match. **Sweep a range** of plausible gap
  profiles, and prefer settings that are robust across all of them.
- Also add position jitter and fragment tracks. Report scores on both the clean and the degraded
  versions.

**Step 6: harness**
- Run any variant on any DFL match and write a `frame,label` CSV.
- Score with `evaluation/possession_metrics.py` unchanged, reporting spellF1, macroF1, possession
  count, mean duration, home share and D share.
- Leave-one-match-out over 7 matches. Grid on 3 matches, then confirm on the other 4.
- Log every run (variant, parameters, match, scores, date). The log feeds the technical document.

**Step 7, optional:** SkillCorner open data as a robustness check (different fps and noise).
Check its licence first.

---

## 6. Selection rule and plausibility checks

**Selection**
- **DFL spellF1 (held-out, degraded) decides.** Plausibility checks can veto.
- Within **~0.04 spellF1** (the noise band from EVALUATION.md), take the **simpler** option.
- A setting must work across all 7 matches, not just the best one. Prefer the centre of a flat
  region over the single best cell.

**Plausibility checks** (run automatically on our match; the team reads them, and Claude only hears
pass/fail)
- Possessions per match, mean and median duration, home share, D share, X share. The **bands come
  from the 7 DFL matches** (min–max plus a margin), stored in `params.py` with their source.
- Structure checks:
  - no impossible switches (faster than a player could reach the ball)
  - `D` runs line up with ball out of bounds or a stationary ball
  - possession changes cluster near close-contact moments
  - all 2nd-half pre-kickoff frames are `D`
- Sensitivity: nudge each parameter and watch counts and durations. A setting that swings wildly is
  rejected.
- A band violation is a **veto to investigate**, never a target to tune toward.

---

## 7. Optional extra: ball trajectory reconstruction

Starts only after A is validated and the DFL harness works. It sits behind a config flag.
- In a gap, predict the ball from its last speed and direction (constant velocity with friction).
  Only trusted for short gaps and ground-like motion, because the data has no ball height.
- Cross-check against players converging on the predicted point and against the next sighting.
  Discard predictions that disagree.
- **Out of play:** if the predicted path crosses a line and the ball isn't seen inside again, mark
  `D`. This has its own parameter and its own test.
- **Honest measurement:** on degraded DFL, the true ball position inside each artificial gap is
  known, so position error and out-of-play accuracy can be scored directly.
- Included only if it beats the noise band on held-out DFL.

**Dropped:** tracklet stitching. Our clues are team-level, so it adds little for a lot of work.

---

## 8. Build order

Each row leaves a working state behind. We go as far as we get today.

| # | step | output | who checks |
|---|---|---|---|
| 1 | tracking loader for our match + ball cleaning (interpolation, speed, out of pitch) | `possession/atd.py`, `ball.py` | Claude on synthetic; team on real |
| 2 | variant A evidence + dead-ball rules + Viterbi decoder + post-filter, with starting values from reasoning | `evidence.py`, `deadball.py`, `hmm.py` | synthetic |
| 3 | `run.py` → CSV, validate, plausibility report | **first valid submission (A)** | team runs, reports in words |
| 4 | DFL licence check, download, loader, labels from per-frame fields | `possession/external/` | Claude on DFL |
| 5 | degradation + harness + leave-one-match-out scoring | run log | Claude on DFL |
| 6 | tune decoder and dead-ball parameters for A on DFL; set the plausibility bands | updated `params.py` | DFL |
| 7 | Smart Tagging factor (step 1, then step 2) | `tagging.py` | synthetic; structure checks |
| 8 | variant B (control events) | B vs A on DFL | DFL |
| 9 | variant C (LightGBM + calibration) | C vs best on DFL | DFL |
| 10 | optional: trajectory reconstruction | flag on/off on DFL | DFL |
| 11 | final run, reproducibility check, public GT validation (once, at the end) | final CSV | team |
| 12 | technical document final, public repo cleaned | deliverables | team |

**Time guard:** if A isn't validated and tuned on DFL by **15 Nov**, stop all extras (B, C,
trajectory) and finish the document and reproducibility.

### Code layout
```
possession/
  atd.py          our tracking → arrays (metres, team per slot)
  ball.py         interpolation, speed, out of pitch
  evidence.py     variants A, B (and C's feature computation)
  deadball.py     dead-ball signals
  tagging.py      Smart Tagging → evidence factor
  hmm.py          Viterbi, transitions, post-filter
  classifier.py   variant C: LightGBM train / calibrate / predict
  external/       DFL loader, labels, degradation, LICENSES.md
  harness.py      leave-one-match-out runs + scoring + run log
  params.py       every value + its source
  run.py          our match → submission CSV + plausibility report
tests/            synthetic matches only
docs/TECHNICAL.md
```
Gitignored: `external_data/`, `possession/out/`, model files that aren't submitted, and every
submission CSV for our match.

---

## 9. Reproducibility and deliverables

**Reproducibility**
- One command: `uv run python -m possession.run --out <path>`. Dependencies pinned in
  `pyproject.toml` / `uv.lock`.
- Fixed seeds; no time- or hardware-dependent steps.
- Learned variant: the model is produced by a documented script on DFL and versioned by hash. If
  it isn't submitted, the variant is off.
- Run twice and compare file hashes. Run once more in a clean environment before submitting.
- Public repo setup is decided later. Before any push, check that there is no `data/`,
  `external_data/`, output folder or our-match CSV in `git status`.

**Technical document** (`docs/TECHNICAL.md`, English, grown as we build)
1. Approach and why: shared decoder, variants A/B/C, which one won on DFL.
2. Assets used and dropped, with reasons (homography, video and broadcast dropped).
3. Training data: DFL/IDSSE, licence, attribution, matches, how labels were derived. None from the
   provided match.
4. Possession rules implemented: loose balls, deflections (`min_hold`), restarts, dead ball.
5. Parameter table: every value and where it came from (DFL fit, literature, reasoning, docs).
6. Limitations: 41% ball missing, identity switches, DFL labels vs the organisers' definition, no
   ball height.
7. Reproduction: command and environment.
8. Compliance statement: no tuning on GT or video, no hand labels, no match data to AI services,
   external data only for training and tuning. The public GT was used once, at the end, as a
   validation check.

**Final checklist (16 Nov)**
- [ ] File validates, and its hash matches a clean rerun
- [ ] Public repo contains no match data and builds from scratch
- [ ] Document covers all 8 points and credits DFL/IDSSE
- [ ] Files named `<team_name>.csv` and `<team_name>.md` or `.pdf`
- [ ] Google Form submitted (final version only)

---

## 10. Decision log (merge of both plans)

| topic | decision | from |
|---|---|---|
| strategy | staged A → B → C on one shared decoder | colleague |
| statistics measured on our match | **dropped**: external + docs only | mine (rules) |
| DFL labels | per-frame ball-status/possession fields first, events as fallback | colleague |
| public GT | only at the end, as validation/plausibility; never tunes | merged |
| organiser email | none; strictest reading on numbers, lineups and GT | team |
| homography | not used | mine |
| classifier (C) | LightGBM + calibration, leave-one-match-out | mine + colleague |
| noise simulation | documented 41% rate, swept gap profiles, jitter, fragmentation | merged |
| extras | trajectory reconstruction yes; tracklet stitching dropped | merged |
| selection | DFL spellF1, 0.04 band → simpler, robust across all 7 matches | colleague |
| plausibility bands | from DFL matches | colleague |
| code folder | `possession/` | mine |
| X | never inside the halves | both |

# Technical document: ball possession, Phase 1

*Draft grown during the build. Numbers marked "DFL" are from external data only. No number here
comes from the provided match, its ground truth or its video.*

## 1. Approach and why
One shared decoder, three interchangeable evidence variants.

- **Decoder.** A Viterbi search over three states (home `H`, away `A`, dead `D`) per half. Each
  frame supplies a score for each state; switching state costs a penalty; afterwards any H/A run
  shorter than `min_hold` (2 s) is merged into its longer neighbour. A frame with no ball evidence
  scores zero for every state, so the decoder keeps its previous state across a gap. The output is
  `H`/`A`/`D` on every frame of the two half blocks and `X` elsewhere.
- **Variant A, geometric.** Evidence from the distance of the nearest home and nearest away player
  to the ball, down-weighted when the ball is fast. Dead ball from hand-written signals (ball out
  of the pitch, ball stationary, players slow and static, the published pre-kickoff block).
- **Variant B, control events.** The team of the last control event (a player within 2 m of a
  slow ball for 0.2 s) holds the ball until the opponent gains control.
- **Variant C, learned.** A gradient-boosted tree classifier (scikit-learn
  `HistGradientBoostingClassifier`, seeded) predicting H/A/D per frame from 34 features: distances
  of the two nearest players of each team, ball speed and turning, distance to the lines,
  team speeds and spread, ball-gap length, and past/future averages over 1 s and 3 s. Trained on
  DFL tracking degraded to resemble ours, probabilities calibrated by one temperature fitted on
  held-out matches. Speeds are computed from positions, not read from the source.
- **Selection.** Held-out DFL spellF1 under degraded conditions decides; within ~0.04 the simpler
  option wins. Result (mean spellF1 over four gap profiles):

| variant | DFL spellF1 |
|---|---|
| A, untuned | 0.16 (tuning matches), 0.20 (confirmation matches) |
| A, tuned | 0.30 / 0.31 |
| B | 0.16 / 0.17 |
| A + trajectory reconstruction | 0.30 / 0.31 (no gain, left off) |
| **C, leave-one-match-out over all 7 matches** | **0.45** (robust across gap lengths and noise levels) |

C is the submitted variant.

## 2. Assets used and dropped
Used: the tactical automatic tracking (players, ball, speeds, team per slot), Smart Tagging
(set pieces as weak dead-ball evidence, the team tag as a mild hint), pitch size and the published
half boundaries and pre-kickoff block. Not used: homography, tactical and broadcast video, the
broadcast tracking, lineups or roles. Reason: the tracking carries the team-level signal we need,
the video must not be used to check output, and the extra sources add alignment work and risk for
little expected gain.

## 3. Training data
**One labelled dataset, none of it from the provided game.**

- **What:** the DFL/IDSSE open data, seven matches (two Bundesliga and five 2. Bundesliga, 2022/23),
  about 1.0 million frames at 25 fps (about 668 minutes) of tracking with per-frame labels.
- **Source and licence:** Figshare doi 10.6084/m9.figshare.28196177, CC BY 4.0, stated both on the
  Figshare record and in the paper ("The present data set is released under a CC-BY 4.0 license with
  the authorization from the DFL and available at figshare"). Credit: Bassek, M., Rein, R.,
  Weber, H. & Memmert, D., *An integrated dataset of spatiotemporal and event data in elite soccer*,
  Scientific Data (2025), doi 10.1038/s41597-025-04505-y.
- **Who labelled it and how:** the labels are the dataset's own per-frame fields, produced by the
  data provider (Sportec Solutions for the DFL) with the TRACAB Gen5 optical tracking system, whose
  output is checked by live operators and in a post-match quality control (paper, Methods and
  Technical Validation). `BallStatus` is 0 while the match is interrupted by the referee, and
  `BallPossession` names the team of the player in control of the ball. We turn them into `D`
  (`BallStatus` 0), `H` and `A`. We did not label anything ourselves.
- **Changes we made:** the tracking is degraded to resemble ours (the ball hidden in contiguous gaps
  on about 41% of frames, position noise, dropped player tracks); speeds are recomputed from
  positions.
- **Use:** training the classifier of variant C, calibrating its probabilities, tuning the decoder,
  and the plausibility bands, all by leave-one-match-out. Hand labels: none, on any game.

## 4. Possession rules implemented
- Loose balls and contested frames follow the evidence; a short run (under 2 s) never counts as a
  possession of its own (`min_hold`), which is how deflections and single touches are absorbed.
- Dead ball: variants A/B use ball out of the pitch (0.5 m margin), ball stationary below 0.3 m/s for
  2 s, players slow and static for 2 s, and the fixed pre-kickoff block. Variant C learns it.
- Restarts: Smart Tagging set-piece windows add weak dead-ball evidence in their first half and a
  mild team hint in their second half.

## 5. Parameters
See `possession/params.py` (every value has a recorded source) and `documentation.md`.

## 6. Limitations
- 41% of ball frames are missing; long gaps are carried, not seen.
- DFL's possession labels flip about every 6 s on average and may follow a different definition
  from the organisers' (deflections, loose balls). The model learns DFL's definition.
- Our tracking is noisier than DFL; degradation is a model of that noise, with unmeasured gap
  lengths, so a range was swept.
- No ball height; a ball in the air can look like a nearby player.
- Slots are tracklets, not people; identities switch.
- All speeds are derived from positions (a central difference over +/-2 frames), identically for DFL
  and our tracking, because the source speed columns differ between the two. Position noise in our
  tracking was not measured; the model was trained and tested over a range of noise levels.

What we would do with more time: measure our own tracking's noise and gap lengths from the tracking
alone (no labels) to tune the degradation to our data; train on more matches and on labels that
follow the challenge's "controlled possession" wording more closely (a first attempt with
event-based labels gave no gain on the DFL data); reduce spurious possession switches and
over-long dead-ball periods; add features for ball height and pass direction.

## 7. Reproduction
Full instructions are in `REPRODUCE.md`. In short: `uv sync`, unpack the data bundle into `data/`
(so that `data/tactical/BAR-ATH_tactical_atd.xml` exists), then
`uv run python -m possession.run --out <file>`. The run uses the committed model
`possession/models/variant_c.joblib` (produced from DFL data by `uv run python -m possession.train_c
final`, checked against `variant_c.sha256` at load time), fixed seeds and pinned dependencies
(`uv.lock`). `uv run python -m possession.check_repro` runs it twice and compares file hashes.
Retraining needs `uv run python -m possession.external.download` first.

## 8. Compliance statement
No tuning on ground truth or video; no hand labels or hand corrections; no match data given to
any AI service; external data only for training, tuning and calibration. The public 5-minute ground
truth was used only as a validation and plausibility check, never to choose a parameter or a variant.
It was scored several times on our final file. The first scoring, on our first output, exposed a
bug (the speed columns of our tracking differed from the training data, so almost every frame was
labelled dead ball); we fixed that mismatch by deriving all speeds from positions and retraining on
the external data only, then scored the corrected file again, including confidence intervals and a
comparison with the organisers' baselines. No parameter was set from those scores.

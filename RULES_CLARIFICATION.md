# Rules clarification from the organisers

> **These rules take precedence over everything else in this repository** — README, RULES.md,
> CHALLENGE.md, DATA.md, FAQ.md, earlier emails and individual answers. Where they disagree, this
> file wins.

Received by email from the organisers, copied verbatim below.

---

Dear participants,

There has been some confusion about what is and what is not permitted in the Initial Challenge, so
we are sending this clarification to all of you.

Please note that the rules below take precedence over any other information you may have received
previously, whether in the documentation, in earlier emails or in answers to individual questions.

### What is not allowed

- Checking your output against the video.
- Hand labels, human-produced data or any other data created manually on the provided game.
- Uploading the provided data to AI services.
- Training, fitting or calibrating your algorithm on the ground-truth possession data or on the
  provided video.

### What is allowed

- Running automatic checks using only the provided data.
- Inspecting the provided data and the video to better understand what counts as possession.
- Using as input the video, the automatic tracking data, the smart tagging or any other generated
  data, as long as your algorithm was not trained on the ground truth or on the provided video.
- Providing files of intermediate steps.
- Using any dataset you have the rights to use. Please check the license of any dataset before
  using it.

### Training and tuning

If you need to train (or retrain), you MUST use only external data. If your algorithm has
parameters that need to be tuned, you MUST tune them on external data, never on the ground-truth
possession data or on the provided video.

If you have any questions, don't hesitate to write to us.

We wish you the best of luck!

---

## What this means for us in practice

Our reading of the email, not the organisers' wording. If in doubt, ask them.

**Inputs we may use:** the tracking data (tactical and broadcast), Smart Tagging, homography and
video, as inputs to a fixed algorithm. External datasets we hold the rights to, for training and
tuning.

**Things we must not do:**

| don't | why |
|---|---|
| Tune any parameter (smoothing windows, distance thresholds, durations…) by looking at the score on `evaluation/gt/BAR-ATH_possession_gt_public.csv` | tuning on the ground truth is forbidden. The old docs called the public GT a "pipeline check"; under this clarification it must not influence any parameter choice |
| Train or fit any model on this match's tracking or video | training is only allowed on external data |
| Watch the video to verify or fix our predicted labels | "checking your output against the video" is forbidden. Watching it to *understand* what possession means is allowed |
| Write or correct any label by hand on this match | no hand labels on the provided game |
| **Paste, upload or send the match data to any AI service** (ChatGPT, Claude, Copilot chat, online notebooks with AI, etc.) | explicitly forbidden. This includes data files, CSV extracts, and script output that prints values from the data. AI assistants may help write *code*, but must not see the data |

**Parameters:** every tunable value needs a justification that doesn't come from this match's
ground truth or video: football knowledge, published literature, or tuning on an external dataset
(for example open tracking data with event labels). Record where each value came from, so the
technical document can explain it.

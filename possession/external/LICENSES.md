# External data licences

## DFL / IDSSE open dataset (training, tuning, validation)

- **Title:** An integrated dataset of spatiotemporal and event data in elite soccer
- **Authors:** Manuel Bassek, Robert Rein, Hendrik Weber, Daniel Memmert
- **Paper:** Scientific Data (2025), doi 10.1038/s41597-025-04505-y
  (https://www.nature.com/articles/s41597-025-04505-y)
- **Data:** Figshare, doi 10.6084/m9.figshare.28196177 (v1, published 2025-02-02)
- **Licence, as stated by Figshare (record `license` field, read 2026-11-07):**
  `CC BY` (Creative Commons Attribution 4.0), https://creativecommons.org/licenses/by/4.0/
- **Licence, as stated in the paper (section "Data Records", read 2026-11-17):**
  "The present data set is released under a CC-BY 4.0 license with the authorization from the DFL
  and available at figshare."
  The paper's article licence (a different statement, about the article text itself) reads: "This
  article is licensed under a Creative Commons Attribution 4.0 International License, which permits
  use, sharing, adaptation, distribution and reproduction in any medium or format, as long as you
  give appropriate credit to the original author(s) and the source, provide a link to the Creative
  Commons licence, and indicate if changes were made."
- **Attribution to use in the technical document:** Bassek, M., Rein, R., Weber, H. & Memmert, D.
  "An integrated dataset of spatiotemporal and event data in elite soccer", Scientific Data (2025),
  doi 10.1038/s41597-025-04505-y; data on Figshare, doi 10.6084/m9.figshare.28196177, CC BY 4.0.
  Changes made: positions were degraded (missing ball, position noise, dropped player tracks) and
  per-frame labels were taken from the data's own ball-status and ball-possession fields; the data
  were used only to train, tune and validate our models.
- **Contents downloaded:** 7 matches (two Bundesliga and five 2. Bundesliga matches, 2022/23):
  match information, raw events, raw observed positions at 25 fps for players, referees and ball.
  The positions carry the ball's own per-frame `BallStatus` (alive/dead) and `BallPossession`
  (home/guest).
- **Storage:** `external_data/idsse/`, gitignored. Never committed. Files verified against the
  md5 published by Figshare (`python -m possession.external.download`).

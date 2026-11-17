# Reproducing the submission

Requires [uv](https://docs.astral.sh/uv/) and the provided data bundle (not in this repository).

```bash
uv sync                                   # pinned environment (pyproject.toml / uv.lock)
# unpack the bundle so these exist (games.py builds the paths; MTAH_DATA_DIR moves the folder):
#   data/tactical/BAR-ATH_tactical_atd.xml
#   data/tactical/BAR-ATH_tactical_atd.txt
#   data/tactical/BAR-ATH_tactical_smart_tagging.xml
uv run python -m possession.run --out Lovania3000.csv
python evaluation/validate_submission.py Lovania3000.csv    # format check only
uv run python -m possession.check_repro                     # two runs, compares file hashes
```

The run reads `possession/models/variant_c.joblib` (trained on DFL/IDSSE open data only; its hash is
in `possession/models/variant_c.sha256` and is checked at load time). It writes a plausibility
report next to the CSV.

## Rebuilding the model (optional, needs internet, about 2.6 GB)

```bash
uv run python -m possession.external.download     # DFL/IDSSE open data, CC BY 4.0, md5-verified
uv run python -m possession.train_c lomo          # leave-one-match-out scores
uv run python -m possession.train_c final         # writes possession/models/variant_c.joblib
```

Tests: `uv run pytest`. Licence and credit for the DFL data: `possession/external/LICENSES.md`.
Approach, limitations and compliance: `docs/TECHNICAL.md`.

Not in this repository (see `.gitignore`): `data/`, `external_data/`, `possession/out/`.

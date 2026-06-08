# Reproducibility Notes

The original data collection queried GPT-3 `davinci` through the legacy OpenAI API. Those calls cannot be faithfully repeated today because the model and service interface are no longer the same reproducibility target. This package therefore treats generated completions and projection scores as archived data.

The replication package stores large raw text/projection tables as compressed CSV files. Run `python scripts/unpack_raw_data.py` before running individual analysis scripts, or run `python scripts/run_all.py --skip-api`, which unpacks them automatically.

The package uses two canonical response/projection CSVs instead of the original hundreds of per-prompt cache files:

- `data/raw/responses/covid_response_projections.csv.gz`
- `data/raw/responses/validation_response_projections.csv.gz`

These files preserve the full generated opinion text in `completion` and the full prompt in `full_prompt`.

Full justification text is included as `data/raw/justifications/justification_texts.csv.gz`, reconstructed from the archived positive/negative justification-list pickle files.

The original justification embedding pickle cache was converted to `data/raw/justifications/justification_embeddings_float32_part*.npz` plus `data/raw/justifications/justification_embeddings_manifest.csv`. The original pickle cache was about 1.8GB, dominated by Python-list embedding pickles. The split float32 NPZ archives are much smaller while retaining enough precision for rerunning exploratory clustering, and each chunk remains below GitHub's 100 MB per-file limit. The manuscript-level justification proportion tests are reproduced from the original cluster counts encoded in `scripts/stat_tests.py`.

The correct/wrong/no-effect forecast coding uses hand-reviewed ground-truth decisions. Most rows can be inferred from `reverse_code`, but school/business open-vs-closed variants require keyword-level handling. `Keeping schools open` is coded as expected liberal, so the final consistency check reports 112 correct, 9 wrong, and 58 no-effect prompt forecasts.

The regenerated Python figures are intended as reproducible analytic figures, not pixel-identical reproductions of edited manuscript graphics.

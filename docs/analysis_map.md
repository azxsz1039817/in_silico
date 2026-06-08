# Analysis Map

| Manuscript / appendix item | Input data | Script | Output |
|---|---|---|---|
| Raw data unpacking | `data/raw/**/*.csv.gz` | `scripts/unpack_raw_data.py` | unpacked local `data/raw/**/*.csv` working files |
| Prompt metadata normalization | `data/raw/prompts/*.csv` | `scripts/build_prompt_tables.py` | `data/processed/*_prompts.csv`, `projection_labels.csv` |
| Main COVID response scoring dataset | unpacked `data/raw/responses/covid_response_projections.csv` | `scripts/assemble_responses.py --dataset covid` | `data/processed/covid_responses.csv`, `covid_responses_named.csv` |
| Validation response scoring dataset | unpacked `data/raw/responses/validation_response_projections.csv` | `scripts/assemble_responses.py --dataset validation` | `data/processed/validation_responses.csv`, `validation_responses_named.csv` |
| COVID coefficient analyses and figures | processed COVID responses | `scripts/coef_plots.py --dataset covid` | COVID coefficient tables and topic-level faceted coefficient figures |
| Validation coefficient analyses and figures | processed validation responses | `scripts/coef_plots.py --dataset validation` | `tables/validation_coefficients.csv`, `figures/validation_coefficients_part*.png` |
| Forecast success binomial tests | manuscript-coded forecast counts | `scripts/stat_tests.py` | `tables/forecast_binomial_tests.csv` |
| Justification cluster proportion tests | manuscript/appendix cluster counts encoded from original R script | `scripts/stat_tests.py` | `tables/justification_two_proportion_tests.csv` |
| Justification bar charts | justification proportion-test table | `scripts/justification_figures.py` | `tables/justification_bar_counts.csv`, `figures/justification_*_bars.png` |
| Forecast consistency check | COVID coefficient table plus prompt metadata | `scripts/validate_consistency.py` | `tables/consistency_checks.csv`, `covid_coefficients_with_consistency_check.csv` |
| Justification cache summary and optional clustering | `data/raw/justifications/justification_cache_summary.csv`, `justification_embeddings_float32_part*.npz`, `justification_texts.csv.gz` | `scripts/cluster_justifications.py` | `data/processed/justification_cache_summary.csv`, optional prompt-level cluster CSVs |
| Optional archival GPT-3 generation provenance | prompt metadata and `OPENAI_API_KEY` | `scripts/archive_generate_gpt3_responses.py`, `scripts/archive_generate_validation_responses.py`, `scripts/archive_generate_justifications.py` | optional rerun outputs, not part of default replication |

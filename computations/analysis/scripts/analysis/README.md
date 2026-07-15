# Claim Evolution Pipeline — Analysis Scripts

This directory contains the full analytical pipeline for the paper
**"From Author's Original to Version of Record: Claim Evolution in AI/ML Preprints"**.

## Pipeline Overview

| Step | Script | Input | Output | Notes |
|------|--------|-------|--------|-------|
| 1 | `run_batch.py` | `claims_extracted.jsonl` | `batch_NN_of_10.jsonl` | LLM annotation (10 batches) |
| 2 | `classify_subfields.py` | `analysis_corpus.csv` | `subfield_labels.csv` | Deterministic, ~2 min |
| 3 | `build_prestige.py` | `analysis_corpus.csv` | `venue_prestige.csv` | Lookup table, ~5 min |
| 4 | `build_dataset.py` | all above | `analysis_dataset.csv`, `pair_level_dataset.csv` | Join + aggregate |
| 5 | `run_models.py` | pair/align datasets | `results/model_results.csv` | Regression models |
| 6 | `make_figures.py` | pair/align datasets | `figures/*.pdf` | Publication figures |

## Quick Start

```bash
# From the project root, with input files in /home/ubuntu/upload/
cd computations/analysis/scripts/analysis

# Step 2: Subfield classification (fast, no LLM)
python3 classify_subfields.py

# Step 3: Venue prestige
python3 build_prestige.py

# Step 4: Build analysis datasets (after all batch files are collected)
python3 build_dataset.py

# Step 5: Fit regression models
pip install statsmodels scikit-learn
python3 run_models.py

# Step 6: Generate all figures
python3 make_figures.py
```

## Environment Variables

All scripts accept path overrides via environment variables:

| Variable | Default | Description |
|----------|---------|-------------|
| `CORPUS_PATH` | `/home/ubuntu/upload/analysis_corpus.csv` | Corpus metadata |
| `CLAIMS_DIR` | `/home/ubuntu/upload` | Directory with batch_NN_of_10.jsonl files |
| `SUBFIELD_PATH` | `/home/ubuntu/upload/subfield_labels.csv` | Step 2 output |
| `PRESTIGE_PATH` | `/home/ubuntu/upload/venue_prestige.csv` | Step 3 output |
| `ALIGN_OUT_PATH` | `/home/ubuntu/upload/analysis_dataset.csv` | Step 4 alignment output |
| `PAIR_OUT_PATH` | `/home/ubuntu/upload/pair_level_dataset.csv` | Step 4 pair output |
| `RESULTS_DIR` | `/home/ubuntu/upload/results` | Step 5 model outputs |
| `FIG_DIR` | `/home/ubuntu/upload/figures` | Step 6 figure outputs |

## Annotation Schema (Step 1)

Each claim alignment record contains:

| Field | Values | Description |
|-------|--------|-------------|
| `semantic` | Unchanged / Clarified / Revised / Removed / Added | Primary change label |
| `scope` | Unchanged / Narrowed / Broadened / N/A | Scope of the claim |
| `confidence` | Unchanged / Tempered / Amplified / N/A | Epistemic confidence |
| `matching_confidence` | 0.0–1.0 | LLM certainty in the alignment |
| `rationale` | string | 1–2 sentence explanation |

## Hypotheses Tested (Step 5)

| Hypothesis | Predictor | Outcome |
|-----------|-----------|---------|
| H1 (Prestige) | `prestige_tier` (1–4) | `pct_tempered` |
| H2 (Subfield) | `subfield` dummies | `pct_changed` |
| H3 (Temporal) | `year_c` (centred 2019) | `pct_changed` |
| H4 (Version) | `version_proxy` | `pct_changed` |
| H5 (Claim type) | claim-level type | `is_changed` |

## Local Project Directory

Save all output files to:
```
<your_project_root>/
  data/
    claims_extracted.jsonl          ← input (from arXiv extraction)
    analysis_corpus.csv             ← input (corpus metadata)
    batch_01_of_10.jsonl            ← Step 1 output (batch 1)
    ...
    batch_10_of_10.jsonl            ← Step 1 output (batch 10)
    subfield_labels.csv             ← Step 2 output
    venue_prestige.csv              ← Step 3 output
    analysis_dataset.csv            ← Step 4 output (alignment-level)
    pair_level_dataset.csv          ← Step 4 output (pair-level)
  results/
    model_results.csv               ← Step 5 output
    ols_pct_changed_summary.txt     ← Step 5 OLS summary
  figures/
    fig1_semantic_distribution.pdf  ← Step 6 output
    ...
    fig8_semantic_confidence_heatmap.pdf
```

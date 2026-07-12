# The Evolution of Scientific Claims from Preprint to Publication in AI and Machine Learning, 2015–2024

**Moses Boudourides**

---

## Overview

This repository contains the computational pipeline and analysis scripts for the paper:

> *The Evolution of Scientific Claims from Preprint to Publication in AI and Machine Learning, 2015–2024*

The study provides the first large-scale quantitative characterisation of scientific claim evolution in AI and machine learning, analysing 51,921 matched preprint–publication pairs drawn from arXiv, bioRxiv, and medRxiv (2015–2024). For each pair, claims are extracted from both the preprint and the published abstract, semantically aligned, and annotated along three independent dimensions: semantic evolution, scope evolution, and confidence evolution.

---

## Annotation Scheme

Claims are annotated along three independent dimensions:

| Dimension | Categories |
|---|---|
| **Semantic evolution** | Unchanged, Clarified, Revised, Removed, Added |
| **Scope evolution** | Unchanged, Narrowed, Broadened |
| **Confidence evolution** | Unchanged, Tempered, Amplified |

Each claim pair receives one label per dimension. Removed and Added claims receive only a Semantic label (Scope and Confidence are not applicable).

---

## Pipeline

1. **Corpus assembly** — arXiv, bioRxiv, medRxiv, and SSRN preprints in AI/ML (2015–2024) matched to their published versions via DOI and metadata linkage.
2. **Claim extraction** — LLM-based extraction of scientific claims from preprint and publication abstracts.
3. **Claim comparison** — LLM-based semantic alignment and multi-dimensional annotation of each claim pair.
4. **Subfield classification** — LLM-based classification of each paper into one of 12 AI/ML subfields.
5. **Venue prestige** — CORE ranking (conferences) and SJR quartile (journals) mapped to a unified 4-point ordinal scale.
6. **Statistical analysis** — Multinomial logistic regression (semantic evolution), ordinal regression (scope and confidence evolution), and citation impact regression.

---

## Requirements

```
python >= 3.11
pandas
numpy
scipy
matplotlib
openai
tqdm
```

Install with:

```bash
pip install pandas numpy scipy matplotlib openai tqdm
```

---

## Data Access

The analysis corpus (51,921 pairs) and claim annotation files are not included in this repository. Publication metadata was collected via the [Dimensions](https://www.dimensions.ai) database. Preprint records were obtained from arXiv, bioRxiv, and medRxiv. The corpus can be reconstructed by running the collection and extraction scripts in `computations/analysis/scripts/collection/` and `computations/analysis/scripts/analysis/` in order, using your own API credentials.

---

## License

MIT License. See [LICENSE](LICENSE).

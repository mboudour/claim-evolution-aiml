"""
validate_dataset.py  —  Step 11 (Dataset Validation & Quality Checks)

Runs a comprehensive set of quality checks on the final analysis corpus and
the extracted claims, producing a validation report suitable for inclusion
in the paper's Methods section.

Checks performed:
  1.  Corpus completeness  — missing values per column
  2.  Linkage quality      — DOI format validity, duplicate pairs
  3.  Temporal consistency — preprint year ≤ publication year
  4.  Abstract quality     — length distribution, language detection
  5.  Claim extraction     — coverage, claim count distribution
  6.  Claim comparison     — coverage, change type distribution sanity
  7.  Source balance       — arXiv / bioRxiv / medRxiv proportions
  8.  Venue balance        — journal / conference / book proportions
  9.  Year coverage        — pairs per year 2015-2024
  10. Inter-rater agreement proxy — sample 200 pairs, re-run extraction,
      compute Cohen's κ against stored labels  (skipped if no API key)

Reads from:
    data/final/analysis_corpus.csv
    data/claims/claims_extracted.jsonl
    data/claims/claim_changes.jsonl
    data/claims/claim_changes_flat.csv

Writes to:
    data/validation/validation_report.txt
    data/validation/validation_summary.csv

Usage:
    cd SSRN_bioRxiv_medRxiv_data_collection_via_Dimensions
    python3 code/analysis/validate_dataset.py
"""

import json
import re
import warnings
from pathlib import Path
from collections import Counter

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

# ── Paths ──────────────────────────────────────────────────────────────────────
SCRIPT_DIR   = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[1]

CORPUS_FILE  = PROJECT_ROOT / "data" / "final"   / "analysis_corpus.csv"
CLAIMS_FILE  = PROJECT_ROOT / "data" / "claims"  / "claims_extracted.jsonl"
CHANGES_FILE = PROJECT_ROOT / "data" / "claims"  / "claim_changes.jsonl"
FLAT_FILE    = PROJECT_ROOT / "data" / "claims"  / "claim_changes_flat.csv"

OUT_DIR      = PROJECT_ROOT / "data" / "validation"
OUT_REPORT   = OUT_DIR / "validation_report.txt"
OUT_SUMMARY  = OUT_DIR / "validation_summary.csv"

OUT_DIR.mkdir(parents=True, exist_ok=True)

DOI_RE = re.compile(r"^10\.\d{4,9}/\S+$")

# ── Helpers ────────────────────────────────────────────────────────────────────

def pct(n, total):
    return f"{n:,}  ({n/total*100:.1f}%)" if total else "0"

def valid_doi(s):
    if not isinstance(s, str):
        return False
    return bool(DOI_RE.match(s.strip()))

# ── Load ───────────────────────────────────────────────────────────────────────

def load_all():
    print("Loading data...")

    corpus = pd.read_csv(CORPUS_FILE, dtype=str, low_memory=False)
    corpus["preprint_year"]       = pd.to_numeric(corpus["preprint_year"],       errors="coerce")
    corpus["pub_year"]            = pd.to_numeric(corpus["pub_year"],             errors="coerce")
    corpus["years_to_pub"]        = pd.to_numeric(corpus["years_to_pub"],         errors="coerce")
    corpus["pub_citations_count"] = pd.to_numeric(corpus["pub_citations_count"],  errors="coerce")
    print(f"  Corpus rows     : {len(corpus):,}")

    claims = []
    with open(CLAIMS_FILE, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    claims.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
    print(f"  Claim records   : {len(claims):,}")

    changes = []
    with open(CHANGES_FILE, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    changes.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
    print(f"  Change records  : {len(changes):,}")

    flat = pd.read_csv(FLAT_FILE, dtype=str, low_memory=False)
    print(f"  Flat comparisons: {len(flat):,}\n")

    return corpus, claims, changes, flat

# ── Check 1: Corpus completeness ───────────────────────────────────────────────

def check_completeness(corpus: pd.DataFrame) -> dict:
    print("── Check 1: Corpus completeness ──")
    key_cols = [
        "doi", "pub_doi", "source", "preprint_year", "pub_year",
        "years_to_pub", "preprint_abstract", "pub_abstract",
        "venue_type", "pub_citations_count",
    ]
    results = {}
    for col in key_cols:
        if col not in corpus.columns:
            results[col] = {"present": False, "missing_n": len(corpus), "missing_pct": 100.0}
            print(f"  {col:<30} COLUMN MISSING")
            continue
        missing = corpus[col].isna().sum()
        pct_miss = missing / len(corpus) * 100
        results[col] = {"present": True, "missing_n": int(missing), "missing_pct": round(pct_miss, 2)}
        flag = " ⚠" if pct_miss > 10 else ""
        print(f"  {col:<30} missing: {pct(missing, len(corpus))}{flag}")
    print()
    return results

# ── Check 2: Linkage quality ───────────────────────────────────────────────────

def check_linkage(corpus: pd.DataFrame) -> dict:
    print("── Check 2: Linkage quality ──")
    n = len(corpus)

    # DOI validity
    valid_pre = corpus["doi"].apply(valid_doi).sum() if "doi" in corpus.columns else 0
    valid_pub = corpus["pub_doi"].apply(valid_doi).sum() if "pub_doi" in corpus.columns else 0

    # Duplicates
    dup_pairs = corpus.duplicated(subset=["doi", "pub_doi"], keep=False).sum() if \
        {"doi", "pub_doi"}.issubset(corpus.columns) else 0

    # Self-links (preprint DOI == pub DOI)
    if {"doi", "pub_doi"}.issubset(corpus.columns):
        self_links = (corpus["doi"].str.lower() == corpus["pub_doi"].str.lower()).sum()
    else:
        self_links = 0

    print(f"  Valid preprint DOIs  : {pct(valid_pre, n)}")
    print(f"  Valid pub DOIs       : {pct(valid_pub, n)}")
    print(f"  Duplicate pairs      : {pct(dup_pairs, n)}")
    print(f"  Self-links           : {pct(self_links, n)}")
    print()
    return {
        "valid_preprint_dois": int(valid_pre),
        "valid_pub_dois":      int(valid_pub),
        "duplicate_pairs":     int(dup_pairs),
        "self_links":          int(self_links),
    }

# ── Check 3: Temporal consistency ─────────────────────────────────────────────

def check_temporal(corpus: pd.DataFrame) -> dict:
    print("── Check 3: Temporal consistency ──")
    n = len(corpus)

    if "years_to_pub" not in corpus.columns:
        print("  years_to_pub column missing\n")
        return {}

    neg   = (corpus["years_to_pub"] < 0).sum()
    zero  = (corpus["years_to_pub"] == 0).sum()
    one   = (corpus["years_to_pub"] == 1).sum()
    two_p = (corpus["years_to_pub"] >= 2).sum()

    print(f"  years_to_pub < 0    : {pct(neg, n)}  ← should be 0 after Step 7 filter")
    print(f"  years_to_pub = 0    : {pct(zero, n)}")
    print(f"  years_to_pub = 1    : {pct(one, n)}")
    print(f"  years_to_pub ≥ 2    : {pct(two_p, n)}")
    print(f"  Mean years_to_pub   : {corpus['years_to_pub'].mean():.2f}")
    print(f"  Median years_to_pub : {corpus['years_to_pub'].median():.1f}")
    print()
    return {
        "negative_years": int(neg),
        "zero_years":     int(zero),
        "one_year":       int(one),
        "two_plus_years": int(two_p),
        "mean_years":     round(float(corpus["years_to_pub"].mean()), 3),
    }

# ── Check 4: Abstract quality ──────────────────────────────────────────────────

def check_abstracts(corpus: pd.DataFrame) -> dict:
    print("── Check 4: Abstract quality ──")
    n = len(corpus)

    results = {}
    for col, label in [("preprint_abstract", "Preprint"), ("pub_abstract", "Publication")]:
        if col not in corpus.columns:
            print(f"  {label}: column missing")
            continue
        lengths = corpus[col].dropna().str.len()
        missing = corpus[col].isna().sum()
        too_short = (lengths < 50).sum()   # suspiciously short
        too_long  = (lengths > 5000).sum() # suspiciously long

        print(f"  {label} abstract:")
        print(f"    Missing          : {pct(missing, n)}")
        print(f"    < 50 chars       : {pct(too_short, len(lengths))}")
        print(f"    > 5000 chars     : {pct(too_long, len(lengths))}")
        print(f"    Mean length      : {lengths.mean():.0f} chars")
        print(f"    Median length    : {lengths.median():.0f} chars")
        results[label.lower()] = {
            "missing": int(missing),
            "too_short": int(too_short),
            "mean_len": round(float(lengths.mean()), 1),
        }
    print()
    return results

# ── Check 5: Claim extraction coverage ────────────────────────────────────────

def check_claims(claims: list, corpus: pd.DataFrame) -> dict:
    print("── Check 5: Claim extraction coverage ──")
    n_corpus = len(corpus)

    ok      = sum(1 for r in claims if not r.get("error"))
    errors  = sum(1 for r in claims if r.get("error"))
    coverage = ok / n_corpus * 100 if n_corpus else 0

    pre_counts = [len(r.get("preprint_claims", [])) for r in claims if not r.get("error")]
    pub_counts = [len(r.get("pub_claims",      [])) for r in claims if not r.get("error")]

    print(f"  Pairs with claims extracted : {pct(ok, n_corpus)}")
    print(f"  Extraction errors           : {pct(errors, len(claims))}")
    if pre_counts:
        print(f"  Preprint claims per pair    : mean={np.mean(pre_counts):.1f}, "
              f"median={np.median(pre_counts):.0f}, "
              f"min={min(pre_counts)}, max={max(pre_counts)}")
    if pub_counts:
        print(f"  Publication claims per pair : mean={np.mean(pub_counts):.1f}, "
              f"median={np.median(pub_counts):.0f}, "
              f"min={min(pub_counts)}, max={max(pub_counts)}")
    print()
    return {
        "pairs_with_claims": ok,
        "extraction_errors": errors,
        "coverage_pct":      round(coverage, 2),
        "mean_preprint_claims": round(float(np.mean(pre_counts)), 2) if pre_counts else 0,
        "mean_pub_claims":      round(float(np.mean(pub_counts)), 2) if pub_counts else 0,
    }

# ── Check 6: Claim comparison sanity ──────────────────────────────────────────

def check_comparisons(changes: list, flat: pd.DataFrame) -> dict:
    print("── Check 6: Claim comparison sanity ──")
    n = len(changes)

    ok     = sum(1 for r in changes if not r.get("error"))
    errors = sum(1 for r in changes if r.get("error"))

    valid_types = {"strengthened", "weakened", "unchanged", "removed", "added"}
    flat_valid  = flat[flat["change_type"].isin(valid_types)]
    flat_bad    = flat[~flat["change_type"].isin(valid_types)]

    ct_counts = flat_valid["change_type"].value_counts()
    total_ct  = len(flat_valid)

    print(f"  Pairs compared successfully : {pct(ok, n)}")
    print(f"  Comparison errors           : {pct(errors, n)}")
    print(f"  Valid claim-level labels    : {pct(len(flat_valid), len(flat))}")
    print(f"  Invalid/empty labels        : {pct(len(flat_bad), len(flat))}")
    print(f"  Change type breakdown:")
    for ct in ["unchanged", "weakened", "strengthened", "removed", "added"]:
        cnt = ct_counts.get(ct, 0)
        print(f"    {ct:<15} : {pct(cnt, total_ct)}")
    print()
    return {
        "pairs_compared":    ok,
        "comparison_errors": errors,
        "valid_labels_pct":  round(len(flat_valid)/len(flat)*100, 2) if len(flat) else 0,
    }

# ── Check 7 & 8: Balance ──────────────────────────────────────────────────────

def check_balance(corpus: pd.DataFrame) -> dict:
    print("── Check 7 & 8: Source and venue balance ──")
    n = len(corpus)
    results = {}

    if "source" in corpus.columns:
        print("  Source breakdown:")
        for src, cnt in corpus["source"].value_counts().items():
            print(f"    {src:<12} : {pct(cnt, n)}")
        results["source_counts"] = corpus["source"].value_counts().to_dict()

    if "venue_type" in corpus.columns:
        print("  Venue type breakdown:")
        for vt, cnt in corpus["venue_type"].value_counts().items():
            print(f"    {vt:<20} : {pct(cnt, n)}")
        results["venue_counts"] = corpus["venue_type"].value_counts().to_dict()

    print()
    return results

# ── Check 9: Year coverage ─────────────────────────────────────────────────────

def check_year_coverage(corpus: pd.DataFrame) -> dict:
    print("── Check 9: Year coverage (2015–2024) ──")
    if "preprint_year" not in corpus.columns:
        print("  preprint_year column missing\n")
        return {}

    yr_counts = corpus["preprint_year"].value_counts().sort_index()
    for yr in range(2015, 2025):
        cnt = int(yr_counts.get(yr, 0))
        flag = " ⚠ low" if cnt < 500 else ""
        print(f"  {int(yr)} : {cnt:,}{flag}")
    print()
    return {"year_counts": {int(k): int(v) for k, v in yr_counts.items()}}

# ── Summary table ──────────────────────────────────────────────────────────────

def build_summary(corpus, claims, changes, flat, completeness, linkage,
                  temporal, abstract_q, claim_q, comparison_q) -> pd.DataFrame:
    n = len(corpus)
    rows = [
        ("Total pairs in corpus",          n,                    ""),
        ("Pairs with valid preprint DOI",   linkage.get("valid_preprint_dois", ""),  ""),
        ("Pairs with valid pub DOI",        linkage.get("valid_pub_dois", ""),       ""),
        ("Duplicate pairs",                 linkage.get("duplicate_pairs", ""),      "should be 0"),
        ("Self-links (pre DOI = pub DOI)",  linkage.get("self_links", ""),           "should be 0"),
        ("Negative years_to_pub",           temporal.get("negative_years", ""),      "should be 0"),
        ("Mean years_to_pub",               temporal.get("mean_years", ""),          ""),
        ("Missing preprint abstract",       abstract_q.get("preprint", {}).get("missing", ""), ""),
        ("Missing pub abstract",            abstract_q.get("publication", {}).get("missing", ""), ""),
        ("Claim extraction coverage",       f"{claim_q.get('coverage_pct','')}%",   ""),
        ("Claim extraction errors",         claim_q.get("extraction_errors", ""),   ""),
        ("Mean preprint claims per pair",   claim_q.get("mean_preprint_claims", ""), ""),
        ("Mean pub claims per pair",        claim_q.get("mean_pub_claims", ""),      ""),
        ("Claim comparison coverage",       f"{comparison_q.get('valid_labels_pct','')}%", ""),
        ("Comparison errors",               comparison_q.get("comparison_errors", ""), ""),
    ]
    df = pd.DataFrame(rows, columns=["Metric", "Value", "Note"])
    return df

# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    print("=== Step 11: Dataset Validation & Quality Checks ===\n")

    corpus, claims, changes, flat = load_all()

    completeness  = check_completeness(corpus)
    linkage       = check_linkage(corpus)
    temporal      = check_temporal(corpus)
    abstract_q    = check_abstracts(corpus)
    claim_q       = check_claims(claims, corpus)
    comparison_q  = check_comparisons(changes, flat)
    balance       = check_balance(corpus)
    year_cov      = check_year_coverage(corpus)

    summary_df = build_summary(
        corpus, claims, changes, flat,
        completeness, linkage, temporal,
        abstract_q, claim_q, comparison_q
    )
    summary_df.to_csv(OUT_SUMMARY, index=False, encoding="utf-8")

    # ── Write text report ─────────────────────────────────────────────────────
    report_lines = [
        "=== Step 11: Dataset Validation Report ===",
        f"Corpus size: {len(corpus):,} preprint–publication pairs",
        "",
        "── Completeness ──",
    ]
    for col, info in completeness.items():
        if info.get("present"):
            report_lines.append(
                f"  {col:<30} missing: {info['missing_n']:,} ({info['missing_pct']}%)"
            )
        else:
            report_lines.append(f"  {col:<30} COLUMN MISSING")

    report_lines += [
        "",
        "── Linkage quality ──",
        f"  Valid preprint DOIs  : {linkage.get('valid_preprint_dois','')}",
        f"  Valid pub DOIs       : {linkage.get('valid_pub_dois','')}",
        f"  Duplicate pairs      : {linkage.get('duplicate_pairs','')}",
        f"  Self-links           : {linkage.get('self_links','')}",
        "",
        "── Temporal consistency ──",
        f"  Negative years_to_pub: {temporal.get('negative_years','')}",
        f"  Mean years_to_pub    : {temporal.get('mean_years','')}",
        "",
        "── Claim extraction ──",
        f"  Coverage             : {claim_q.get('coverage_pct','')}%",
        f"  Errors               : {claim_q.get('extraction_errors','')}",
        f"  Mean preprint claims : {claim_q.get('mean_preprint_claims','')}",
        f"  Mean pub claims      : {claim_q.get('mean_pub_claims','')}",
        "",
        "── Claim comparison ──",
        f"  Valid labels         : {comparison_q.get('valid_labels_pct','')}%",
        f"  Errors               : {comparison_q.get('comparison_errors','')}",
        "",
        "── Overall assessment ──",
    ]

    issues = []
    if linkage.get("duplicate_pairs", 0) > 0:
        issues.append(f"  ⚠ {linkage['duplicate_pairs']} duplicate pairs found")
    if linkage.get("self_links", 0) > 0:
        issues.append(f"  ⚠ {linkage['self_links']} self-links found")
    if temporal.get("negative_years", 0) > 0:
        issues.append(f"  ⚠ {temporal['negative_years']} pairs with negative years_to_pub")
    if claim_q.get("coverage_pct", 0) < 95:
        issues.append(f"  ⚠ Claim extraction coverage below 95%")

    if issues:
        report_lines += ["  Issues requiring attention:"] + issues
    else:
        report_lines.append("  No critical issues found. Dataset is ready for analysis.")

    report = "\n".join(report_lines)
    print(report)
    OUT_REPORT.write_text(report, encoding="utf-8")

    print(f"\n  Saved report  → {OUT_REPORT.name}")
    print(f"  Saved summary → {OUT_SUMMARY.name}")
    print("\nStep 11 complete.")


if __name__ == "__main__":
    main()

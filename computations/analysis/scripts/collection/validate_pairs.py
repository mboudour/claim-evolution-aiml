"""
validate_pairs.py  —  Step 5 (Pair validation)

Uses the preprint_title already present in the cached BigQuery links file
(data/linked_pairs/bq_preprint_links/preprint_links_bq.csv) to validate
matched pairs by title similarity. No additional BigQuery queries needed.

Filters applied:
  1. Title similarity — keeps pairs where the preprint title in our corpus
     matches the Dimensions preprint title (fuzzy token sort ratio >= 60).
     A low score indicates a wrong match (different paper linked by coincidence).
  2. Abstract availability — flags pairs missing a preprint abstract
     (does NOT remove them; Step 6 will retrieve missing abstracts).

Reads from:
    data/linked_pairs/doi_linked/pairs_doi_linked.csv
    data/linked_pairs/bq_preprint_links/preprint_links_bq.csv

Writes to:
    data/validated/pairs_validated.csv      <- pairs passing all filters
    data/validated/pairs_rejected.csv       <- pairs failing any filter
    data/validated/validation_report.txt    <- summary statistics

Usage:
    cd SSRN_bioRxiv_medRxiv_data_collection_via_Dimensions
    python3 code/collection/validate_pairs.py

Requirements:
    pip install pandas rapidfuzz tqdm
"""

import re
import pandas as pd
from pathlib import Path
from tqdm import tqdm
from rapidfuzz import fuzz

# ── Paths ──────────────────────────────────────────────────────────────────────
SCRIPT_DIR   = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[1]

PAIRS_FILE = PROJECT_ROOT / "data" / "linked_pairs" / "doi_linked" / "pairs_doi_linked.csv"
LINKS_FILE = PROJECT_ROOT / "data" / "linked_pairs" / "bq_preprint_links" / "preprint_links_bq.csv"

OUT_DIR    = PROJECT_ROOT / "data" / "validated"
OUT_VALID  = OUT_DIR / "pairs_validated.csv"
OUT_REJECT = OUT_DIR / "pairs_rejected.csv"
OUT_REPORT = OUT_DIR / "validation_report.txt"

OUT_DIR.mkdir(parents=True, exist_ok=True)

# ── Settings ───────────────────────────────────────────────────────────────────
TITLE_SIM_THRESHOLD = 60   # minimum fuzzy ratio (0–100) to keep a pair

# ── Helpers ────────────────────────────────────────────────────────────────────

def clean_title(t):
    if not t or pd.isna(t):
        return ""
    t = str(t).lower()
    t = re.sub(r"[^\w\s]", " ", t)
    t = re.sub(r"\s+", " ", t).strip()
    return t

def title_similarity(t1, t2):
    t1 = clean_title(t1)
    t2 = clean_title(t2)
    if not t1 or not t2:
        return 0.0
    return fuzz.token_sort_ratio(t1, t2)

# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    print("=== Step 5: Pair Validation ===\n")

    # Load pairs
    print("Loading pairs...")
    pairs = pd.read_csv(PAIRS_FILE, dtype=str, low_memory=False)
    print(f"  Pairs: {len(pairs):,}")

    # Load links (contains preprint_title from Dimensions)
    print("Loading Dimensions preprint titles...")
    links = pd.read_csv(LINKS_FILE, dtype=str, low_memory=False)
    print(f"  Links: {len(links):,}\n")

    # Join preprint_title onto pairs via pub_doi (the shared key)
    # Each pair has a pub_doi; the links file has pub_doi_norm and preprint_title
    pairs["pub_doi_norm_key"] = pairs["pub_doi"].fillna("").str.strip().str.lower()
    links["pub_doi_norm_key"] = links["pub_doi_norm"].fillna("").str.strip().str.lower()

    # Also join via arxiv_id_norm for arXiv pairs
    pairs_with_title = pairs.merge(
        links[["pub_doi_norm_key", "preprint_title"]].drop_duplicates("pub_doi_norm_key"),
        on="pub_doi_norm_key",
        how="left"
    )

    n_total = len(pairs_with_title)
    n_title_found = pairs_with_title["preprint_title"].notna().sum()
    print(f"  Pairs with Dimensions title available: {n_title_found:,} / {n_total:,}\n")

    # ── Title similarity ───────────────────────────────────────────────────────
    print("Computing title similarity scores...")
    corpus_titles    = pairs_with_title["title"].fillna("").tolist()
    dimensions_titles = pairs_with_title["preprint_title"].fillna("").tolist()

    scores = [
        title_similarity(ct, dt)
        for ct, dt in tqdm(zip(corpus_titles, dimensions_titles),
                           total=n_total, desc="  Similarity")
    ]
    pairs_with_title["title_similarity"] = scores

    # Pairs with no Dimensions title get score=0 → keep them (benefit of doubt)
    # Only reject pairs where BOTH titles are present but dissimilar
    pairs_with_title["flag_title"] = (
        (pairs_with_title["preprint_title"].notna()) &
        (pairs_with_title["title_similarity"] < TITLE_SIM_THRESHOLD)
    )

    n_rejected = pairs_with_title["flag_title"].sum()
    n_valid    = n_total - n_rejected

    print(f"\n  Title similarity filter (threshold={TITLE_SIM_THRESHOLD}):")
    print(f"    Pairs rejected : {n_rejected:,}")
    print(f"    Pairs kept     : {n_valid:,}  ({n_valid/n_total*100:.1f}%)")
    print(f"    Mean score     : {pairs_with_title['title_similarity'].mean():.1f}")
    print(f"    Median score   : {pairs_with_title['title_similarity'].median():.1f}")

    # ── Abstract availability ──────────────────────────────────────────────────
    pairs_with_title["has_preprint_abstract"] = (
        pairs_with_title["abstract"].fillna("").str.strip().str.len() > 20
    )
    n_missing_abstract = (~pairs_with_title["has_preprint_abstract"]).sum()
    print(f"\n  Preprint abstract availability:")
    print(f"    Missing abstract : {n_missing_abstract:,}")

    # ── Split and save ─────────────────────────────────────────────────────────
    valid   = pairs_with_title[~pairs_with_title["flag_title"]].copy()
    rejected = pairs_with_title[pairs_with_title["flag_title"]].copy()
    rejected["rejection_reason"] = "title_similarity"

    valid.to_csv(OUT_VALID,   index=False, encoding="utf-8")
    rejected.to_csv(OUT_REJECT, index=False, encoding="utf-8")

    # ── Report ─────────────────────────────────────────────────────────────────
    report_lines = [
        "=== Step 5: Pair Validation Report ===\n",
        f"Total pairs input                    : {n_total:,}",
        f"Pairs with Dimensions title          : {n_title_found:,}",
        f"Pairs rejected (title similarity)    : {n_rejected:,}",
        f"Pairs validated                      : {n_valid:,}  ({n_valid/n_total*100:.1f}%)",
        f"\nTitle similarity stats (all pairs):",
        f"  Mean   : {pairs_with_title['title_similarity'].mean():.1f}",
        f"  Median : {pairs_with_title['title_similarity'].median():.1f}",
        f"  Min    : {pairs_with_title['title_similarity'].min():.1f}",
        f"  Max    : {pairs_with_title['title_similarity'].max():.1f}",
        f"\nAbstract availability (validated pairs):",
        f"  Missing preprint abstract : {(~valid['has_preprint_abstract']).sum():,}",
        f"\nLinkage method breakdown (validated):",
        valid["linkage_method"].value_counts().to_string(),
        f"\nYear breakdown (validated pairs, preprint year):",
        valid["year"].value_counts().sort_index().to_string(),
        f"\nSource breakdown (validated pairs):",
        valid["source"].value_counts().to_string(),
    ]
    report = "\n".join(report_lines)
    print("\n" + report)
    OUT_REPORT.write_text(report, encoding="utf-8")

    print(f"\nOutput files:")
    print(f"  Validated : {OUT_VALID}")
    print(f"  Rejected  : {OUT_REJECT}")
    print(f"  Report    : {OUT_REPORT}")
    print("\nStep 5 complete.")


if __name__ == "__main__":
    main()

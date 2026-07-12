"""
enrich_metadata.py  —  Step 7 (Metadata enrichment)

Produces the final analysis-ready dataset by:

  1. Computing time-to-publication (days from preprint date to publication year)
  2. Classifying publication venue type (journal article / conference / book chapter)
  3. Flagging open-access status
  4. Normalising field-of-research categories
  5. Filtering to pairs with BOTH abstracts present (the analysis corpus)
  6. Writing a clean final dataset with well-named columns

Reads from:
    data/validated/pairs_enriched.csv

Writes to:
    data/final/analysis_corpus.csv          <- analysis-ready dataset
    data/final/corpus_stats.txt             <- summary statistics

Usage:
    cd SSRN_bioRxiv_medRxiv_data_collection_via_Dimensions
    python3 code/collection/enrich_metadata.py

Requirements:
    pip install pandas tqdm
"""

import pandas as pd
from pathlib import Path
from datetime import datetime

# ── Paths ──────────────────────────────────────────────────────────────────────
SCRIPT_DIR   = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[1]

INPUT_FILE  = PROJECT_ROOT / "data" / "validated" / "pairs_enriched.csv"
OUT_DIR     = PROJECT_ROOT / "data" / "final"
OUT_CORPUS  = OUT_DIR / "analysis_corpus.csv"
OUT_STATS   = OUT_DIR / "corpus_stats.txt"

OUT_DIR.mkdir(parents=True, exist_ok=True)

# ── Venue type mapping ─────────────────────────────────────────────────────────
VENUE_MAP = {
    "article":    "journal_article",
    "proceeding": "conference_paper",
    "chapter":    "book_chapter",
    "preprint":   "preprint",
    "book":       "book",
    "monograph":  "book",
}

# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    print("=== Step 7: Metadata Enrichment ===\n")

    print("Loading enriched pairs...")
    df = pd.read_csv(INPUT_FILE, dtype=str, low_memory=False)
    print(f"  Total pairs loaded: {len(df):,}\n")

    # ── 1. Filter to pairs with both abstracts ─────────────────────────────────
    has_pre_abs = df["abstract"].fillna("").str.strip().str.len() > 20
    has_pub_abs = df["pub_abstract"].fillna("").str.strip().str.len() > 20
    df_both = df[has_pre_abs & has_pub_abs].copy()
    print(f"  Pairs with both abstracts: {len(df_both):,}")
    n_dropped = len(df) - len(df_both)
    print(f"  Dropped (missing abstract): {n_dropped:,}\n")

    # ── 2. Venue type ──────────────────────────────────────────────────────────
    df_both["venue_type"] = (
        df_both["pub_type"].fillna("").str.strip().str.lower()
        .map(VENUE_MAP)
        .fillna("other")
    )

    # ── 3. Time to publication ─────────────────────────────────────────────────
    # preprint date is in 'date' column (YYYY-MM-DD or YYYY)
    # publication year is in 'pub_year' column
    def parse_year(s):
        try:
            return int(str(s).strip()[:4])
        except Exception:
            return None

    def parse_date_year(s):
        try:
            return int(str(s).strip()[:4])
        except Exception:
            return None

    df_both["preprint_year_int"] = df_both["year"].apply(parse_year)
    df_both["pub_year_int"]      = df_both["pub_year"].apply(parse_year)
    df_both["years_to_pub"]      = df_both["pub_year_int"] - df_both["preprint_year_int"]

    # ── 3b. Filter negative years_to_pub ───────────────────────────────────────────
    # Negative values indicate wrong matches (pub year before preprint year)
    n_before = len(df_both)
    df_both = df_both[df_both["years_to_pub"].fillna(-1).astype(float) >= 0].copy()
    n_negative = n_before - len(df_both)
    print(f"  Removed {n_negative:,} pairs with negative time-to-publication.")
    print(f"  Remaining pairs: {len(df_both):,}\n")

    # ── 4. Open access flag ────────────────────────────────────────────────────
    # Dimensions returns OA category strings like 'oa_all', 'gold', 'green', etc.
    # Non-OA publications have empty or 'closed' values.
    oa_values = df_both["pub_open_access"].fillna("").str.strip().str.lower()
    df_both["is_open_access"] = (
        oa_values.str.contains(r"gold|green|hybrid|bronze|oa", regex=True, na=False)
        & ~oa_values.isin(["", "closed", "not_oa"])
    )

    # ── 5. Field of research ───────────────────────────────────────────────────
    df_both["field_of_research_clean"] = (
        df_both["field_of_research"].fillna("")
        .str.replace(r"\[|\]|'", "", regex=True)
        .str.strip()
    )

    # ── 6. Build final clean dataset ──────────────────────────────────────────
    final = df_both[[
        # Identifiers
        "source", "arxiv_id", "doi", "pub_doi",
        "dimensions_id", "pub_dimensions_id",
        # Preprint metadata
        "title", "abstract", "authors", "year", "date",
        "categories", "journal_ref",
        "field_of_research_clean",
        "research_org_names", "research_org_countries",
        # Publication metadata
        "pub_title", "pub_abstract", "pub_year", "pub_year_int",
        "pub_type", "venue_type",
        "pub_journal", "pub_journal_id",
        "pub_publisher", "pub_open_access", "is_open_access",
        "pub_citations_count",
        "pub_research_orgs", "pub_research_countries",
        # Linkage
        "linkage_method", "title_similarity",
        "preprint_year_int", "years_to_pub",
    ]].copy()

    final = final.rename(columns={
        "title":                  "preprint_title",
        "abstract":               "preprint_abstract",
        "authors":                "preprint_authors",
        "year":                   "preprint_year",
        "date":                   "preprint_date",
        "categories":             "preprint_categories",
        "journal_ref":            "preprint_journal_ref",
        "field_of_research_clean":"field_of_research",
        "research_org_names":     "preprint_org_names",
        "research_org_countries": "preprint_org_countries",
    })

    final.to_csv(OUT_CORPUS, index=False, encoding="utf-8")
    print(f"  Saved analysis corpus → {OUT_CORPUS.name}")

    # ── 7. Summary statistics ──────────────────────────────────────────────────
    lines = [
        "=== Step 7: Analysis Corpus Statistics ===\n",
        f"Total pairs in analysis corpus     : {len(final):,}",
        f"\nSource breakdown:",
        final["source"].value_counts().to_string(),
        f"\nVenue type breakdown:",
        final["venue_type"].value_counts().to_string(),
        f"\nLinkage method breakdown:",
        final["linkage_method"].value_counts().to_string(),
        f"\nPreprint year breakdown:",
        final["preprint_year"].value_counts().sort_index().to_string(),
        f"\nTime to publication (years) — distribution:",
        final["years_to_pub"].value_counts().sort_index().to_string(),
        f"\nOpen access:",
        f"  Open access publications: {final['is_open_access'].sum():,}  "
        f"({final['is_open_access'].mean()*100:.1f}%)",
        f"\nAbstract length stats:",
        f"  Preprint abstract (chars) — mean: {final['preprint_abstract'].str.len().mean():.0f}, "
        f"median: {final['preprint_abstract'].str.len().median():.0f}",
        f"  Publication abstract (chars) — mean: {final['pub_abstract'].str.len().mean():.0f}, "
        f"median: {final['pub_abstract'].str.len().median():.0f}",
    ]
    stats = "\n".join(lines)
    print("\n" + stats)
    OUT_STATS.write_text(stats, encoding="utf-8")

    print(f"\n  Saved corpus stats → {OUT_STATS.name}")
    print("\nStep 7 complete.")
    print(f"\nFinal analysis corpus: {len(final):,} preprint–publication pairs")
    print(f"Output: {OUT_CORPUS}")


if __name__ == "__main__":
    main()

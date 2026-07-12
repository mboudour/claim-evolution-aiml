"""
deduplicate.py  —  Step 3

Merge the four preprint source files and deduplicate by DOI and arXiv ID.

Input files:
    data/raw/arxiv/processed/arxiv_aiml_2015_2024.csv
    data/raw/biorxiv/raw_export/biorxiv_aiml_2015_2024.csv
    data/raw/medrxiv/raw_export/medrxiv_aiml_2015_2024.csv
    data/raw/ssrn/raw_export/ssrn_aiml_2015_2024.csv

Output:
    data/deduplicated/preprints_aiml_2015_2024.csv

Deduplication logic:
    1. Normalise DOI (lowercase, strip whitespace).
    2. Normalise arXiv ID (strip 'arxiv:' prefix, lowercase).
    3. Within arXiv records: deduplicate on arXiv ID (keeps first occurrence,
       which is the earliest category listing).
    4. Across all sources: deduplicate on normalised DOI (keeps arXiv record
       when both arXiv and a Dimensions server have the same DOI).
    5. Records with no DOI and no arXiv ID are kept as-is (cannot be matched).

Usage:
    python3 code/collection/deduplicate.py

Requirements:
    pip install pandas tqdm
"""

import re
import pandas as pd
from pathlib import Path
from tqdm import tqdm

# ── Paths ──────────────────────────────────────────────────────────────────────

BASE_DIR = Path(__file__).resolve().parents[2]

SOURCES = {
    "arxiv":   BASE_DIR / "data" / "raw"  / "arxiv"   / "processed"  / "arxiv_aiml_2015_2024.csv",
    "biorxiv": BASE_DIR / "data" / "raw"  / "biorxiv"  / "raw_export" / "biorxiv_aiml_2015_2024.csv",
    "medrxiv": BASE_DIR / "data" / "raw"  / "medrxiv"  / "raw_export" / "medrxiv_aiml_2015_2024.csv",
    "ssrn":    BASE_DIR / "data" / "raw"  / "ssrn"     / "raw_export" / "ssrn_aiml_2015_2024.csv",
}

OUTPUT_DIR  = BASE_DIR / "data" / "deduplicated"
OUTPUT_FILE = OUTPUT_DIR / "preprints_aiml_2015_2024.csv"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ── Normalisation helpers ──────────────────────────────────────────────────────

def norm_doi(doi):
    """Lowercase, strip whitespace and common URL prefixes."""
    if pd.isna(doi) or str(doi).strip() == "":
        return ""
    doi = str(doi).strip().lower()
    doi = re.sub(r"^https?://(dx\.)?doi\.org/", "", doi)
    return doi

def norm_arxiv_id(arxiv_id):
    """Strip 'arxiv:' prefix and version suffix, lowercase."""
    if pd.isna(arxiv_id) or str(arxiv_id).strip() == "":
        return ""
    aid = str(arxiv_id).strip().lower()
    aid = re.sub(r"^arxiv:", "", aid)
    aid = re.sub(r"v\d+$", "", aid)   # strip version suffix e.g. 2301.00001v2 → 2301.00001
    return aid

# ── Load and harmonise each source ────────────────────────────────────────────

def load_arxiv(path: Path) -> pd.DataFrame:
    print(f"  Loading arXiv from {path.name} ...")
    df = pd.read_csv(path, dtype=str, low_memory=False)
    df["source"] = "arxiv"
    df["arxiv_id_norm"] = df["arxiv_id"].apply(norm_arxiv_id)
    df["doi_norm"]      = df["doi"].apply(norm_doi)
    # arXiv records do not have resulting_publication_doi in this file
    df["resulting_publication_doi"] = ""
    # Rename for unified schema
    df = df.rename(columns={
        "first_submission_date": "date",
        "first_submission_year": "year",
    })
    keep = ["source", "arxiv_id_norm", "doi_norm", "arxiv_id",
            "title", "abstract", "authors", "year", "date",
            "doi", "categories", "journal_ref",
            "resulting_publication_doi"]
    return df[[c for c in keep if c in df.columns]]


def load_dimensions(path: Path, source_name: str) -> pd.DataFrame:
    print(f"  Loading {source_name} from {path.name} ...")
    df = pd.read_csv(path, dtype=str, low_memory=False)
    df["source"] = source_name
    df["doi_norm"]      = df["doi"].apply(norm_doi)
    df["arxiv_id_norm"] = ""   # Dimensions bioRxiv/medRxiv/SSRN records have no arXiv ID
    keep = ["source", "arxiv_id_norm", "doi_norm",
            "dimensions_id", "title", "abstract", "authors",
            "year", "date", "doi", "source_title", "publisher",
            "resulting_publication_doi", "open_access",
            "field_of_research", "research_org_names", "research_org_countries"]
    return df[[c for c in keep if c in df.columns]]


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    print("=== Step 3: Deduplication ===\n")

    # 1. Load all sources
    frames = []
    for name, path in SOURCES.items():
        if not path.exists():
            print(f"  WARNING: {path} not found — skipping.")
            continue
        if name == "arxiv":
            frames.append(load_arxiv(path))
        else:
            frames.append(load_dimensions(path, name))

    print(f"\nRecords before deduplication:")
    for f in frames:
        src = f["source"].iloc[0]
        print(f"  {src:10s}: {len(f):>8,}")

    combined = pd.concat(frames, ignore_index=True, sort=False)
    total_raw = len(combined)
    print(f"  {'TOTAL':10s}: {total_raw:>8,}\n")

    # 2. Deduplicate arXiv internally on arXiv ID
    #    (same paper cross-listed to multiple categories appears once per category)
    arxiv_mask = (combined["source"] == "arxiv") & (combined["arxiv_id_norm"] != "")
    arxiv_df   = combined[arxiv_mask].drop_duplicates(subset="arxiv_id_norm", keep="first")
    non_arxiv  = combined[~arxiv_mask]
    combined   = pd.concat([arxiv_df, non_arxiv], ignore_index=True)
    print(f"After arXiv internal dedup (by arXiv ID):  {len(combined):>8,}")

    # 3. Deduplicate across sources on DOI
    #    Priority: arxiv > biorxiv > medrxiv > ssrn
    #    (arXiv record is preferred when DOI matches a Dimensions record)
    source_priority = {"arxiv": 0, "biorxiv": 1, "medrxiv": 2, "ssrn": 3}
    combined["_priority"] = combined["source"].map(source_priority).fillna(9)
    combined = combined.sort_values("_priority")

    has_doi  = combined["doi_norm"] != ""
    no_doi   = combined[~has_doi].copy()
    with_doi = combined[has_doi].drop_duplicates(subset="doi_norm", keep="first").copy()

    combined = pd.concat([with_doi, no_doi], ignore_index=True)
    combined = combined.drop(columns=["_priority"])
    print(f"After cross-source dedup (by DOI):         {len(combined):>8,}")

    # 4. Report
    print(f"\nFinal deduplicated corpus: {len(combined):,} records")
    print(f"  Removed: {total_raw - len(combined):,} duplicates\n")

    print("Source breakdown after deduplication:")
    print(combined["source"].value_counts().to_string())

    print("\nYear breakdown after deduplication:")
    combined["year"] = pd.to_numeric(combined["year"], errors="coerce")
    print(combined["year"].value_counts().sort_index().to_string())

    print(f"\nRecords with resulting_publication_doi: "
          f"{(combined['resulting_publication_doi'].fillna('') != '').sum():,}")

    # 5. Save
    combined.to_csv(OUTPUT_FILE, index=False, encoding="utf-8")
    print(f"\nSaved → {OUTPUT_FILE}")
    print("Step 3 complete.")


if __name__ == "__main__":
    main()

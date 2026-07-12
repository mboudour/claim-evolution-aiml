"""
query_bigquery.py  —  Step 4 (BigQuery approach)

Strategy:
  The Dimensions BigQuery `publications` table stores preprint records
  (arXiv, bioRxiv, medRxiv, SSRN) with a `resulting_publication_doi` field
  that points to the corresponding journal publication.

  Phase 1 (BigQuery): Pull all preprint records (2015-2024) that have a
  `resulting_publication_doi`. This gives us the preprint→journal DOI mapping.

  Phase 2 (local): Join the BigQuery results against our preprint corpus
  (preprints_aiml_2015_2024.csv) by DOI and arXiv ID to produce matched pairs.

Prerequisites (run once in your terminal):
    pip install google-cloud-bigquery pandas pyarrow tqdm db-dtypes
    gcloud auth application-default login
    gcloud auth application-default set-quota-project my-first-project-156507

Usage:
    cd SSRN_bioRxiv_medRxiv_data_collection_via_Dimensions
    python3 code/collection/query_bigquery.py

Output:
    data/linked_pairs/bq_preprint_links/preprint_links_bq.csv   ← Phase 1
    data/linked_pairs/doi_linked/pairs_doi_linked.csv            ← Phase 2
    data/linked_pairs/unmatched/preprints_unmatched.csv
"""

import pandas as pd
from pathlib import Path
from google.cloud import bigquery

# ── Paths ──────────────────────────────────────────────────────────────────────
SCRIPT_DIR   = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[1]

PREPRINTS_FILE = PROJECT_ROOT / "data" / "deduplicated" / "preprints_aiml_2015_2024.csv"

LINKS_DIR  = PROJECT_ROOT / "data" / "linked_pairs" / "bq_preprint_links"
LINKS_FILE = LINKS_DIR / "preprint_links_bq.csv"

OUT_PAIRS     = PROJECT_ROOT / "data" / "linked_pairs" / "doi_linked"  / "pairs_doi_linked.csv"
OUT_UNMATCHED = PROJECT_ROOT / "data" / "linked_pairs" / "unmatched"   / "preprints_unmatched.csv"

for p in [LINKS_FILE, OUT_PAIRS, OUT_UNMATCHED]:
    p.parent.mkdir(parents=True, exist_ok=True)

# ── Configuration ──────────────────────────────────────────────────────────────
GCP_PROJECT_ID = "my-first-project-156507"
DATASET        = "dimensions_on_gbq"
TABLE          = "publications"

# ── Phase 1 Query ──────────────────────────────────────────────────────────────
# Pull all preprint records that have a resulting_publication_doi.
# We keep: the preprint DOI, the arXiv ID (if any), and the journal DOI.
QUERY = f"""
SELECT
    id                                  AS preprint_dimensions_id,
    doi                                 AS preprint_doi,
    arxiv_id,
    resulting_publication_doi           AS pub_doi,
    year,
    date,
    title.preferred                     AS preprint_title
FROM `{GCP_PROJECT_ID}.{DATASET}.{TABLE}`
WHERE
    year BETWEEN 2015 AND 2024
    AND resulting_publication_doi IS NOT NULL
"""

# ── Normalise helpers ──────────────────────────────────────────────────────────

def norm_doi(doi):
    doi = (doi or "").strip().lower()
    for prefix in ["https://doi.org/", "http://doi.org/",
                   "https://dx.doi.org/", "http://dx.doi.org/"]:
        doi = doi.replace(prefix, "")
    return doi.split(";")[0].strip()

def norm_arxiv_id(aid):
    aid = (aid or "").strip().lower()
    # Remove common prefixes
    for prefix in ["arxiv:", "http://arxiv.org/abs/", "https://arxiv.org/abs/"]:
        aid = aid.replace(prefix, "")
    aid = aid.strip()
    # Remove version suffix e.g. v1, v2
    if aid and "v" in aid:
        parts = aid.rsplit("v", 1)
        if len(parts) == 2 and parts[1].isdigit():
            aid = parts[0]
    return aid

# ── Phase 1: Download preprint→publication links from BigQuery ─────────────────

def download_links():
    print("── Phase 1: Downloading preprint→publication links from BigQuery ──\n")
    print(f"Table  : {GCP_PROJECT_ID}.{DATASET}.{TABLE}")
    print(f"Output : {LINKS_FILE}\n")

    client = bigquery.Client(project=GCP_PROJECT_ID)

    print("Submitting query...")
    job = client.query(QUERY)

    print("Waiting for results (this may take a few minutes)...")
    df = job.to_dataframe(progress_bar_type="tqdm")

    print(f"\nDownloaded {len(df):,} preprint→publication links.")

    # Normalise keys
    df["arxiv_id_norm"] = df["arxiv_id"].apply(norm_arxiv_id)
    df["preprint_doi_norm"] = df["preprint_doi"].apply(norm_doi)
    df["pub_doi_norm"] = df["pub_doi"].apply(norm_doi)

    df.to_csv(LINKS_FILE, index=False, encoding="utf-8")
    print(f"Saved → {LINKS_FILE}\n")
    return df

# ── Phase 2: Local join ────────────────────────────────────────────────────────

def join_pairs(links_df):
    print("── Phase 2: Local join against preprint corpus ──\n")

    print("  Loading preprints...")
    preprints = pd.read_csv(PREPRINTS_FILE, dtype=str, low_memory=False)
    print(f"  Preprints: {len(preprints):,}")

    # Normalise preprint join keys
    # Build a unique row key from arxiv_id_norm + doi (the file has no 'id' column)
    preprints["arxiv_id_norm"] = preprints["arxiv_id"].fillna("").apply(norm_arxiv_id)
    preprints["doi_norm"]      = preprints["doi"].fillna("").apply(norm_doi)
    preprints["_row_key"] = preprints["arxiv_id_norm"].where(
        preprints["arxiv_id_norm"] != "",
        preprints["doi_norm"]
    )

    # ── Join 1: arXiv preprints matched by arXiv ID ────────────────────────────
    arxiv_links = links_df[links_df["arxiv_id_norm"].str.strip() != ""].copy()
    arxiv_pre   = preprints[
        (preprints["source"] == "arxiv") &
        (preprints["arxiv_id_norm"].str.strip() != "")
    ].copy()

    arxiv_matched = arxiv_pre.merge(
        arxiv_links[["arxiv_id_norm", "pub_doi", "pub_doi_norm",
                     "preprint_dimensions_id", "preprint_doi"]],
        on="arxiv_id_norm", how="inner"
    )
    arxiv_matched["linkage_method"] = "arxiv_id"
    print(f"  arXiv pairs matched   : {len(arxiv_matched):,}")

    # ── Join 2: bioRxiv/medRxiv/SSRN matched by preprint DOI ──────────────────
    other_pre = preprints[
        preprints["source"].isin(["biorxiv", "medrxiv", "ssrn"]) &
        (preprints["doi_norm"].str.strip() != "")
    ].copy()

    doi_links = links_df[links_df["preprint_doi_norm"].str.strip() != ""].copy()

    doi_matched = other_pre.merge(
        doi_links[["preprint_doi_norm", "pub_doi", "pub_doi_norm",
                   "preprint_dimensions_id"]],
        left_on="doi_norm", right_on="preprint_doi_norm",
        how="inner"
    )
    doi_matched["linkage_method"] = "preprint_doi"
    print(f"  Other server pairs    : {len(doi_matched):,}")

    # ── Combine ────────────────────────────────────────────────────────────────
    all_pairs = pd.concat([arxiv_matched, doi_matched], ignore_index=True)
    all_pairs = all_pairs.drop_duplicates(subset=["_row_key"], keep="first")
    all_pairs.to_csv(OUT_PAIRS, index=False, encoding="utf-8")
    print(f"\n  Total matched pairs   : {len(all_pairs):,} → {OUT_PAIRS.name}")

    # ── Unmatched ──────────────────────────────────────────────────────────────
    matched_keys = set(all_pairs["_row_key"].fillna(""))
    unmatched = preprints[~preprints["_row_key"].isin(matched_keys)]
    unmatched.to_csv(OUT_UNMATCHED, index=False, encoding="utf-8")
    print(f"  Unmatched preprints   : {len(unmatched):,} → {OUT_UNMATCHED.name}")

    # ── Summary ────────────────────────────────────────────────────────────────
    print(f"\n── Summary ──")
    print(f"  Total preprints       : {len(preprints):,}")
    print(f"  Matched pairs         : {len(all_pairs):,}")
    print(f"  Match rate            : {len(all_pairs)/len(preprints)*100:.1f}%")
    print(f"\n  Linkage method breakdown:")
    print(all_pairs["linkage_method"].value_counts().to_string())
    print(f"\n  Year breakdown (preprint year):")
    print(all_pairs["year"].value_counts().sort_index().to_string())

# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    print("=== Step 4: Publication Linkage via BigQuery ===\n")

    # Phase 1: download links from BigQuery (or reload if already done)
    if LINKS_FILE.exists():
        print(f"  Found existing links file: {LINKS_FILE}")
        print("  Loading cached links...")
        links_df = pd.read_csv(LINKS_FILE, dtype=str, low_memory=False)
        links_df["arxiv_id_norm"]    = links_df["arxiv_id_norm"].fillna("")
        links_df["preprint_doi_norm"] = links_df["preprint_doi_norm"].fillna("")
        links_df["pub_doi_norm"]     = links_df["pub_doi_norm"].fillna("")
        print(f"  Loaded {len(links_df):,} links.\n")
    else:
        links_df = download_links()

    # Phase 2: local join
    join_pairs(links_df)

    print("\nStep 4 complete.")


if __name__ == "__main__":
    main()

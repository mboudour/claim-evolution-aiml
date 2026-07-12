"""
fetch_pub_abstracts.py  —  Step 6 (Publication abstract retrieval)

Fetches publication metadata (abstract, title, journal, document type,
citations count, open access status) from the Dimensions API using the
pub_doi values in the validated pairs file.

Strategy:
  - Queries Dimensions DSL in batches of 100 DOIs per request
  - 61,495 pairs → ~615 API requests → ~10–15 minutes
  - Results are saved incrementally so the script is resumable

Reads from:
    data/validated/pairs_validated.csv
    config/dimensions_key.txt

Writes to:
    data/validated/pub_abstracts_raw.csv     ← raw API results (cached)
    data/validated/pairs_enriched.csv        ← final enriched pairs file

Usage:
    cd SSRN_bioRxiv_medRxiv_data_collection_via_Dimensions
    python3 code/collection/fetch_pub_abstracts.py

Requirements:
    pip install requests pandas tqdm
"""

import time
import requests
import pandas as pd
from pathlib import Path
from tqdm import tqdm

# ── Paths ──────────────────────────────────────────────────────────────────────
SCRIPT_DIR   = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[1]

PAIRS_FILE   = PROJECT_ROOT / "data" / "validated" / "pairs_validated.csv"
KEY_FILE     = PROJECT_ROOT / "config" / "dimensions_key.txt"
OUT_RAW      = PROJECT_ROOT / "data" / "validated" / "pub_abstracts_raw.csv"
OUT_ENRICHED = PROJECT_ROOT / "data" / "validated" / "pairs_enriched.csv"

# ── Configuration ──────────────────────────────────────────────────────────────
DIMENSIONS_AUTH_URL = "https://app.dimensions.ai/api/auth.json"
DIMENSIONS_DSL_URL  = "https://app.dimensions.ai/api/dsl.json"
BATCH_SIZE          = 100    # DOIs per request (Dimensions DSL limit)
RETRY_WAIT          = 10     # seconds to wait on rate-limit error
MAX_RETRIES         = 3

# Fields to retrieve for each publication (Dimensions DSL field names)
RETURN_FIELDS = (
    "id+doi+title+abstract+year+type"
    "+journal+open_access"
    "+times_cited+publisher"
    "+research_org_names+research_org_countries"
)

# ── Authentication ─────────────────────────────────────────────────────────────

def get_token() -> str:
    key = KEY_FILE.read_text().strip()
    if not key:
        raise ValueError(f"{KEY_FILE} is empty.")
    resp = requests.post(DIMENSIONS_AUTH_URL, json={"key": key}, timeout=30)
    resp.raise_for_status()
    token = resp.json().get("token")
    if not token:
        raise ValueError("Authentication failed — check your Dimensions API key.")
    print("  Authenticated with Dimensions API.")
    return token

# ── DSL query for a batch of DOIs ──────────────────────────────────────────────

def build_query(dois: list) -> str:
    # Format DOI list as DSL array: ["doi1","doi2",...]
    # Strip backslashes and double-quotes — neither is valid in a DOI
    def clean_doi(d):
        return d.replace('\\', '').replace('"', '')
    doi_list = '["' + '","'.join(clean_doi(d) for d in dois) + '"]'
    return (
        f'search publications\n'
        f'where doi in {doi_list}\n'
        f'return publications[{RETURN_FIELDS}]\n'
        f'limit {BATCH_SIZE}'
    )

# ── Fetch one batch ────────────────────────────────────────────────────────────

def fetch_batch(dois: list, headers: dict) -> list:
    query = build_query(dois)
    for attempt in range(MAX_RETRIES):
        try:
            resp = requests.post(
                DIMENSIONS_DSL_URL,
                data=query,
                headers=headers,
                timeout=60
            )
            if resp.status_code == 200:
                return resp.json().get("publications", [])
            elif resp.status_code in (429, 503):
                wait = RETRY_WAIT * (attempt + 1)
                tqdm.write(f"  Rate limited — waiting {wait}s...")
                time.sleep(wait)
            else:
                tqdm.write(f"  HTTP {resp.status_code}: {resp.text[:200]}")
                return []
        except requests.exceptions.RequestException as e:
            tqdm.write(f"  Request error: {e}")
            time.sleep(RETRY_WAIT)
    return []

# ── Flatten one publication record ─────────────────────────────────────────────

def flatten_pub(pub: dict) -> dict:
    journal = pub.get("journal") or {}
    return {
        "pub_dimensions_id":    pub.get("id", ""),
        "pub_doi_raw":          pub.get("doi", ""),
        "pub_title":            pub.get("title", ""),
        "pub_abstract":         pub.get("abstract", ""),
        "pub_year":             pub.get("year", ""),
        "pub_type":             pub.get("type", ""),
        "pub_journal":          journal.get("title", ""),
        "pub_journal_id":       journal.get("id", ""),
        "pub_open_access":      pub.get("open_access", ""),
        "pub_citations_count":  pub.get("times_cited", ""),
        "pub_publisher":        str(pub.get("publisher") or ""),
        "pub_research_orgs":    "; ".join(
            str(x.get("name", x) if isinstance(x, dict) else x)
            for x in (pub.get("research_org_names") or [])
        ),
        "pub_research_countries": "; ".join(
            str(x.get("name", x) if isinstance(x, dict) else x)
            for x in (pub.get("research_org_countries") or [])
        ),
    }

# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    print("=== Step 6: Publication Abstract Retrieval (Dimensions API) ===\n")

    # Load validated pairs
    print("Loading validated pairs...")
    pairs = pd.read_csv(PAIRS_FILE, dtype=str, low_memory=False)
    print(f"  Pairs: {len(pairs):,}")

    all_pub_dois = pairs["pub_doi"].dropna().str.strip().str.lower().unique().tolist()
    print(f"  Unique publication DOIs: {len(all_pub_dois):,}\n")

    # Check for cached results
    already_fetched = set()
    raw_records = []
    if OUT_RAW.exists():
        cached = pd.read_csv(OUT_RAW, dtype=str, low_memory=False)
        raw_records = cached.to_dict("records")
        already_fetched = set(cached["pub_doi_raw"].str.strip().str.lower().tolist())
        print(f"  Resuming — {len(already_fetched):,} DOIs already fetched.\n")

    # Filter to DOIs not yet fetched
    remaining_dois = [d for d in all_pub_dois if d not in already_fetched]
    print(f"  DOIs to fetch: {len(remaining_dois):,}")

    if remaining_dois:
        print("  Authenticating...")
        token   = get_token()
        headers = {"Authorization": f"JWT {token}"}

        batches = [remaining_dois[i:i+BATCH_SIZE]
                   for i in range(0, len(remaining_dois), BATCH_SIZE)]

        print(f"  Fetching {len(batches):,} batches of up to {BATCH_SIZE} DOIs each...\n")

        for batch in tqdm(batches, desc="  Fetching"):
            pubs = fetch_batch(batch, headers)
            for pub in pubs:
                raw_records.append(flatten_pub(pub))
            # Small sleep to be polite to the API
            time.sleep(0.5)

        # Save raw results
        raw_df = pd.DataFrame(raw_records)
        raw_df.to_csv(OUT_RAW, index=False, encoding="utf-8")
        print(f"\n  Saved {len(raw_df):,} publication records → {OUT_RAW.name}")
    else:
        print("  All DOIs already fetched. Loading from cache...\n")
        raw_df = pd.DataFrame(raw_records)

    # ── Merge onto validated pairs ─────────────────────────────────────────────
    print("\nMerging publication metadata onto validated pairs...")
    raw_df["pub_doi_norm"] = raw_df["pub_doi_raw"].str.strip().str.lower()
    pairs["pub_doi_norm"]  = pairs["pub_doi"].fillna("").str.strip().str.lower()

    enriched = pairs.merge(
        raw_df.drop_duplicates("pub_doi_norm"),
        on="pub_doi_norm",
        how="left"
    )

    n_with_abstract = enriched["pub_abstract"].fillna("").str.strip().str.len().gt(20).sum()
    n_missing       = len(enriched) - n_with_abstract

    print(f"  Total pairs              : {len(enriched):,}")
    print(f"  With publication abstract: {n_with_abstract:,}")
    print(f"  Missing pub abstract     : {n_missing:,}")

    enriched.to_csv(OUT_ENRICHED, index=False, encoding="utf-8")
    print(f"\n  Saved enriched pairs → {OUT_ENRICHED.name}")

    # ── Summary ────────────────────────────────────────────────────────────────
    print(f"\n── Summary ──")
    print(f"  Pairs with both preprint AND publication abstract: "
          f"{(enriched['abstract'].fillna('').str.len().gt(20) & enriched['pub_abstract'].fillna('').str.len().gt(20)).sum():,}")
    print(f"\n  Publication type breakdown:")
    print(enriched["pub_type"].value_counts().head(10).to_string())
    print(f"\n  Year breakdown (preprint year):")
    print(enriched["year"].value_counts().sort_index().to_string())

    print("\nStep 6 complete.")


if __name__ == "__main__":
    main()

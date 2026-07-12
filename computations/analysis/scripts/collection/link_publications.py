"""
link_publications.py  —  Step 4 (inverted bulk download, monthly stratification)

Strategy:
  1. Query Dimensions month by month (Jan 2015 – Dec 2024 = 120 sub-queries),
     downloading all published papers that have an arXiv ID and a non-arXiv DOI.
     Each monthly sub-query returns at most ~15,000 records, safely under the
     Dimensions 50,000-record ceiling.
  2. Join the downloaded published papers against the preprint corpus locally
     by normalised arXiv ID (for arXiv records) and by resulting_publication_doi
     (for bioRxiv/medRxiv/SSRN records).

Estimated runtime: ~80 minutes (120 months × ~40 seconds per month).
Fully resumable: completed months are recorded in checkpoint_months.txt.

Place this script at:
    SSRN_bioRxiv_medRxiv_data_collection_via_Dimensions/code/collection/link_publications.py

Reads from:
    data/deduplicated/preprints_aiml_2015_2024.csv
    config/dimensions_key.txt

Writes to:
    data/linked_pairs/published_with_arxiv/published_arxiv_2015_2024.csv
    data/linked_pairs/published_with_arxiv/checkpoint_months.txt
    data/linked_pairs/doi_linked/pairs_doi_linked.csv
    data/linked_pairs/unmatched/preprints_unmatched.csv

Usage:
    cd SSRN_bioRxiv_medRxiv_data_collection_via_Dimensions
    python3 code/collection/link_publications.py

Requirements:
    pip install requests pandas tqdm
"""

import time
import calendar
import requests
import pandas as pd
from pathlib import Path
from tqdm import tqdm

# ── Paths ──────────────────────────────────────────────────────────────────────

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT    = SCRIPT_DIR.parents[1]

KEY_FILE   = PROJECT / "config" / "dimensions_key.txt"
INPUT_FILE = PROJECT / "data" / "deduplicated" / "preprints_aiml_2015_2024.csv"

PUB_DIR    = PROJECT / "data" / "linked_pairs" / "published_with_arxiv"
PUB_FILE   = PUB_DIR / "published_arxiv_2015_2024.csv"
CHECKPOINT = PUB_DIR / "checkpoint_months.txt"

OUT_PAIRS     = PROJECT / "data" / "linked_pairs" / "doi_linked"  / "pairs_doi_linked.csv"
OUT_UNMATCHED = PROJECT / "data" / "linked_pairs" / "unmatched"   / "preprints_unmatched.csv"

for p in [PUB_FILE, CHECKPOINT, OUT_PAIRS, OUT_UNMATCHED]:
    p.parent.mkdir(parents=True, exist_ok=True)

# ── API settings ───────────────────────────────────────────────────────────────

DIMENSIONS_AUTH_URL = "https://app.dimensions.ai/api/auth.json"
DIMENSIONS_DSL_URL  = "https://app.dimensions.ai/api/dsl.json"

PAGE_SIZE     = 1000   # records per page (Dimensions max)
SLEEP_BETWEEN = 0.5    # seconds between requests
FLUSH_EVERY   = 20000  # records before flushing to disk

PUB_FIELDS = (
    "id+title+abstract+doi+arxiv_id+year+date"
    "+journal+source_title+publisher"
    "+document_type+proceedings_title"
    "+open_access+field_citation_ratio"
    "+times_cited+recent_citations"
    "+category_for"
    "+research_org_names+research_org_countries"
)

# ── Authentication ─────────────────────────────────────────────────────────────

def get_token() -> str:
    key = KEY_FILE.read_text().strip()
    resp = requests.post(DIMENSIONS_AUTH_URL, json={"key": key}, timeout=30)
    resp.raise_for_status()
    token = resp.json().get("token")
    if not token:
        raise ValueError("Authentication failed — check config/dimensions_key.txt")
    return token

# ── Flatten a published record ─────────────────────────────────────────────────

def flatten_pub(pub: dict) -> dict:
    journal = pub.get("journal") or {}
    journal_title = journal.get("title", "") if isinstance(journal, dict) else str(journal)
    journal_id    = journal.get("id", "")    if isinstance(journal, dict) else ""

    cats = pub.get("category_for", []) or []
    cat_str = "; ".join(c.get("name", "") for c in cats if isinstance(c, dict))

    oa = pub.get("open_access", []) or []
    oa_str = "; ".join(str(o) for o in oa) if isinstance(oa, list) else str(oa)

    countries = pub.get("research_org_countries", []) or []
    country_str = "; ".join(
        c.get("name", "") if isinstance(c, dict) else str(c) for c in countries
    )
    orgs = pub.get("research_org_names", []) or []
    orgs_str = "; ".join(str(o) for o in orgs)

    return {
        "pub_dimensions_id":          pub.get("id", ""),
        "pub_title":                  (pub.get("title", "") or "").replace("\n", " ").strip(),
        "pub_abstract":               (pub.get("abstract", "") or "").replace("\n", " ").strip(),
        "pub_doi":                    pub.get("doi", ""),
        "pub_arxiv_id":               pub.get("arxiv_id", ""),
        "pub_year":                   pub.get("year", ""),
        "pub_date":                   pub.get("date", ""),
        "pub_journal_title":          journal_title,
        "pub_journal_id":             journal_id,
        "pub_source_title":           pub.get("source_title", ""),
        "pub_publisher":              pub.get("publisher", ""),
        "pub_document_type":          pub.get("document_type", ""),
        "pub_proceedings_title":      pub.get("proceedings_title", ""),
        "pub_open_access":            oa_str,
        "pub_field_citation_ratio":   pub.get("field_citation_ratio", ""),
        "pub_times_cited":            pub.get("times_cited", ""),
        "pub_recent_citations":       pub.get("recent_citations", ""),
        "pub_field_of_research":      cat_str,
        "pub_research_org_names":     orgs_str,
        "pub_research_org_countries": country_str,
    }

# ── Normalise arXiv ID ─────────────────────────────────────────────────────────

def norm_arxiv_id(aid: str) -> str:
    aid = (aid or "").strip().lower()
    aid = aid.replace("arxiv:", "").strip()
    # Remove version suffix e.g. v1, v2
    if aid and "v" in aid:
        parts = aid.rsplit("v", 1)
        if len(parts) == 2 and parts[1].isdigit():
            aid = parts[0]
    return aid

# ── Normalise DOI ──────────────────────────────────────────────────────────────

def norm_doi(doi: str) -> str:
    doi = (doi or "").strip().lower()
    for prefix in ["https://doi.org/", "http://doi.org/",
                    "https://dx.doi.org/", "http://dx.doi.org/"]:
        doi = doi.replace(prefix, "")
    return doi.split(";")[0].strip()

# ── Phase 1: Download all published papers with arXiv IDs ─────────────────────

def download_published(token: str):
    print("── Phase 1: Downloading published papers with arXiv IDs ──\n")

    # Load checkpoint: set of "YYYY-MM" strings already completed
    done_months = set()
    if CHECKPOINT.exists():
        done_months = set(l.strip() for l in CHECKPOINT.read_text().splitlines() if l.strip())
        if done_months:
            print(f"  Resuming — {len(done_months)} months already completed.")

    write_header = not PUB_FILE.exists()
    records_buf  = []

    # Generate all year-month combinations 2015-01 through 2024-12
    months = []
    for year in range(2015, 2025):
        for month in range(1, 13):
            months.append((year, month))

    for year, month in tqdm(months, desc="Months", unit="month"):
        ym = f"{year}-{month:02d}"
        if ym in done_months:
            continue

        # Date range for this month
        last_day = calendar.monthrange(year, month)[1]
        date_from = f"{year}-{month:02d}-01"
        date_to   = f"{year}-{month:02d}-{last_day:02d}"

        # Get total count for this month
        count_q = (
            f'search publications\n'
            f'where date >= "{date_from}"\n'
            f'and date <= "{date_to}"\n'
            f'and arxiv_id is not empty\n'
            f'and doi is not empty\n'
            f'return publications[id]\n'
            f'limit 1 skip 0'
        )
        time.sleep(SLEEP_BETWEEN)
        resp = requests.post(
            DIMENSIONS_DSL_URL, data=count_q,
            headers={"Authorization": f"JWT {token}"}, timeout=60
        )
        if resp.status_code != 200:
            tqdm.write(f"  {ym}: ERROR getting count — {resp.text[:150]}")
            continue

        total = resp.json().get("_stats", {}).get("total_count", 0)
        if total == 0:
            with open(CHECKPOINT, "a") as f:
                f.write(f"{ym}\n")
            continue

        if total > 49000:
            tqdm.write(f"  WARNING: {ym} has {total:,} records — approaching 50K ceiling!")

        pages = (total + PAGE_SIZE - 1) // PAGE_SIZE

        for page in range(pages):
            skip = page * PAGE_SIZE
            q = (
                f'search publications\n'
                f'where date >= "{date_from}"\n'
                f'and date <= "{date_to}"\n'
                f'and arxiv_id is not empty\n'
                f'and doi is not empty\n'
                f'return publications[{PUB_FIELDS}]\n'
                f'limit {PAGE_SIZE} skip {skip}'
            )
            time.sleep(SLEEP_BETWEEN)
            r = requests.post(
                DIMENSIONS_DSL_URL, data=q,
                headers={"Authorization": f"JWT {token}"}, timeout=120
            )
            if r.status_code != 200:
                tqdm.write(f"  {ym} page {page}: ERROR — {r.text[:150]}")
                continue

            pubs = r.json().get("publications", [])
            for pub in pubs:
                flat = flatten_pub(pub)
                flat["arxiv_id_norm"] = norm_arxiv_id(flat["pub_arxiv_id"])
                # Exclude records whose DOI is an arXiv DOI (preprints, not publications)
                pub_doi = (flat.get("pub_doi") or "").lower()
                if pub_doi.startswith("10.48550"):
                    continue
                records_buf.append(flat)

        # Flush buffer
        if len(records_buf) >= FLUSH_EVERY:
            pd.DataFrame(records_buf).to_csv(
                PUB_FILE, mode="a", header=write_header,
                index=False, encoding="utf-8"
            )
            write_header = False
            records_buf = []

        # Mark month as done
        with open(CHECKPOINT, "a") as f:
            f.write(f"{ym}\n")

    # Final flush
    if records_buf:
        pd.DataFrame(records_buf).to_csv(
            PUB_FILE, mode="a", header=write_header,
            index=False, encoding="utf-8"
        )

    total_downloaded = sum(1 for _ in open(PUB_FILE)) - 1 if PUB_FILE.exists() else 0
    print(f"\n  Total published records downloaded: {total_downloaded:,}")
    print(f"  Saved → {PUB_FILE.name}\n")

# ── Phase 2: Local join ────────────────────────────────────────────────────────

def join_pairs():
    print("── Phase 2: Local join ──\n")

    print("  Loading preprints...")
    preprints = pd.read_csv(INPUT_FILE, dtype=str, low_memory=False)
    print(f"  Preprints: {len(preprints):,}")

    print("  Loading published papers...")
    published = pd.read_csv(PUB_FILE, dtype=str, low_memory=False)
    print(f"  Published (with arXiv ID, non-arXiv DOI): {len(published):,}")

    # Normalise join keys
    preprints["arxiv_id_join"] = preprints["arxiv_id_norm"].fillna("").apply(norm_arxiv_id)
    published["arxiv_id_join"] = published["arxiv_id_norm"].fillna("").apply(norm_arxiv_id)

    # ── Join 1: arXiv preprints → published papers by arXiv ID ────────────────
    arxiv_pre = preprints[
        (preprints["source"] == "arxiv") &
        (preprints["arxiv_id_join"].str.strip() != "")
    ].copy()

    arxiv_matched = arxiv_pre.merge(
        published[published["arxiv_id_join"].str.strip() != ""],
        on="arxiv_id_join", how="inner", suffixes=("", "_pub")
    )
    arxiv_matched["linkage_method"] = "arxiv_id_join"
    print(f"\n  arXiv pairs matched   : {len(arxiv_matched):,}")

    # ── Join 2: bioRxiv/medRxiv/SSRN → published papers by DOI ───────────────
    other_pre = preprints[
        preprints["source"].isin(["biorxiv", "medrxiv", "ssrn"]) &
        (preprints["resulting_publication_doi"].fillna("").str.strip() != "")
    ].copy()
    other_pre["rpd_norm"] = other_pre["resulting_publication_doi"].apply(norm_doi)
    published["pub_doi_norm"] = published["pub_doi"].apply(norm_doi)

    other_matched = other_pre.merge(
        published[published["pub_doi_norm"].str.strip() != ""],
        left_on="rpd_norm", right_on="pub_doi_norm",
        how="inner", suffixes=("", "_pub")
    )
    other_matched["linkage_method"] = "dimensions_native_doi"
    print(f"  Other server pairs    : {len(other_matched):,}")

    # ── Combine ────────────────────────────────────────────────────────────────
    pub_cols = [c for c in arxiv_matched.columns if c.startswith("pub_") or c == "linkage_method"]
    pre_cols = list(preprints.columns)
    keep_cols = pre_cols + [c for c in pub_cols if c not in pre_cols]

    all_pairs = pd.concat([
        arxiv_matched[[c for c in keep_cols if c in arxiv_matched.columns]],
        other_matched[[c for c in keep_cols if c in other_matched.columns]]
    ], ignore_index=True)

    # One pair per preprint
    all_pairs = all_pairs.drop_duplicates(subset=["id"], keep="first")
    all_pairs.to_csv(OUT_PAIRS, index=False, encoding="utf-8")
    print(f"\n  Total matched pairs   : {len(all_pairs):,} → {OUT_PAIRS.name}")

    # ── Unmatched ──────────────────────────────────────────────────────────────
    matched_ids = set(all_pairs["id"].fillna(""))
    unmatched = preprints[~preprints["id"].isin(matched_ids)]
    unmatched.to_csv(OUT_UNMATCHED, index=False, encoding="utf-8")
    print(f"  Unmatched preprints   : {len(unmatched):,} → {OUT_UNMATCHED.name}")

    # ── Summary ────────────────────────────────────────────────────────────────
    print(f"\n── Summary ──")
    print(f"  Total preprints       : {len(preprints):,}")
    print(f"  Matched pairs         : {len(all_pairs):,}")
    print(f"  Match rate            : {len(all_pairs)/len(preprints)*100:.1f}%")
    print(f"\n  Linkage method breakdown:")
    print(all_pairs["linkage_method"].value_counts().to_string())
    print(f"\n  Document type breakdown (top 10):")
    print(all_pairs["pub_document_type"].value_counts().head(10).to_string())
    print(f"\n  Year breakdown (preprint year):")
    print(all_pairs["year"].value_counts().sort_index().to_string())

# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    print("=== Step 4: Publication Linkage (monthly stratification) ===\n")
    print(f"Project root : {PROJECT}")
    print(f"Input file   : {INPUT_FILE}\n")

    token = get_token()
    print("Authenticated with Dimensions API.\n")

    download_published(token)
    join_pairs()

    print("\nStep 4 complete.")


if __name__ == "__main__":
    main()

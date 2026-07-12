"""
query_dimensions.py

Query the Dimensions API for AI/ML preprints from bioRxiv, medRxiv, and SSRN
for the period 2015-2024.

Verified Dimensions source title IDs:
    bioRxiv  = jour.1293558
    medRxiv  = jour.1369542
    SSRN     = jour.1276748

Key linkage field: resulting_publication_doi
(Dimensions natively stores the published DOI for each preprint)

Reads the Dimensions API key from: config/dimensions_key.txt
Outputs one CSV per server into:
    data/raw/biorxiv/raw_export/biorxiv_aiml_2015_2024.csv
    data/raw/medrxiv/raw_export/medrxiv_aiml_2015_2024.csv
    data/raw/ssrn/raw_export/ssrn_aiml_2015_2024.csv

Usage:
    python3 code/collection/query_dimensions.py

Requirements:
    pip install requests pandas tqdm
"""

import time
import requests
import pandas as pd
from tqdm import tqdm
from pathlib import Path

# ── Configuration ─────────────────────────────────────────────────────────────

BASE_DIR = Path(__file__).resolve().parents[2]
KEY_FILE = BASE_DIR / "config" / "dimensions_key.txt"

DIMENSIONS_AUTH_URL = "https://app.dimensions.ai/api/auth.json"
DIMENSIONS_DSL_URL  = "https://app.dimensions.ai/api/dsl.json"

START_YEAR = 2015
END_YEAR   = 2024
PAGE_SIZE  = 1000

# Verified Dimensions source title IDs
SERVERS = {
    "biorxiv": {
        "source_id": "jour.1293558",
        "label":     "bioRxiv",
        "output":    BASE_DIR / "data" / "raw" / "biorxiv" / "raw_export" / "biorxiv_aiml_2015_2024.csv",
    },
    "medrxiv": {
        "source_id": "jour.1369542",
        "label":     "medRxiv",
        "output":    BASE_DIR / "data" / "raw" / "medrxiv" / "raw_export" / "medrxiv_aiml_2015_2024.csv",
    },
    "ssrn": {
        "source_id": "jour.1276748",
        "label":     "SSRN Electronic Journal",
        "output":    BASE_DIR / "data" / "raw" / "ssrn" / "raw_export" / "ssrn_aiml_2015_2024.csv",
    },
}

# Fields to return — note: + separator, NOT commas
RETURN_FIELDS = (
    "id+title+abstract+authors+year+date+doi"
    "+source_title+publisher"
    "+resulting_publication_doi"
    "+open_access"
    "+category_for"
    "+research_org_names+research_org_countries"
)

# AI/ML keyword filter — unquoted terms joined with OR (no phrase quotes needed)
# Phrase searches use escaped quotes inside the for clause
AIML_FOR = (
    r'\"machine learning\" OR \"deep learning\" OR \"neural network\" OR '
    r'\"artificial intelligence\" OR \"large language model\" OR '
    r'\"natural language processing\" OR \"computer vision\" OR '
    r'\"reinforcement learning\" OR \"transformer\" OR \"generative model\" OR '
    r'\"foundation model\" OR \"diffusion model\" OR \"graph neural\" OR '
    r'\"federated learning\" OR \"self-supervised\"'
)

# ── Authentication ─────────────────────────────────────────────────────────────

def get_token(key_file: Path) -> str:
    key = key_file.read_text().strip()
    if not key:
        raise ValueError(f"{key_file} is empty.")
    resp = requests.post(DIMENSIONS_AUTH_URL, json={"key": key}, timeout=30)
    resp.raise_for_status()
    token = resp.json().get("token")
    if not token:
        raise ValueError("Authentication failed — check your Dimensions API key.")
    print("Authenticated with Dimensions API.")
    return token


# ── DSL Query Builder ──────────────────────────────────────────────────────────

def build_query(source_id: str, skip: int) -> str:
    return (
        f'search publications\n'
        f'in title_abstract_only\n'
        f'for "{AIML_FOR}"\n'
        f'where source_title.id = "{source_id}"\n'
        f'and year in [{START_YEAR}:{END_YEAR}]\n'
        f'return publications[{RETURN_FIELDS}]\n'
        f'limit {PAGE_SIZE} skip {skip}'
    )


# ── Paginated Fetcher ──────────────────────────────────────────────────────────

def fetch_all(label: str, source_id: str, token: str) -> list:
    headers = {"Authorization": f"JWT {token}"}
    records = []
    skip = 0

    print(f"\n── Querying: {label} (id={source_id}) ──")

    resp = requests.post(
        DIMENSIONS_DSL_URL,
        data=build_query(source_id, skip=0),
        headers=headers,
        timeout=120
    )

    if resp.status_code != 200:
        print(f"  ERROR {resp.status_code}: {resp.text[:500]}")
        return []

    data  = resp.json()
    total = data.get("_stats", {}).get("total_count", 0)
    batch = data.get("publications", [])
    print(f"  Total records available: {total:,}")

    records.extend(batch)
    skip += len(batch)

    with tqdm(total=total, initial=len(batch), unit="rec") as pbar:
        while skip < total:
            time.sleep(0.5)
            resp = requests.post(
                DIMENSIONS_DSL_URL,
                data=build_query(source_id, skip=skip),
                headers=headers,
                timeout=120
            )
            if resp.status_code != 200:
                print(f"\n  ERROR {resp.status_code} at skip={skip}: {resp.text[:300]}")
                break
            batch = resp.json().get("publications", [])
            if not batch:
                break
            records.extend(batch)
            skip += len(batch)
            pbar.update(len(batch))

    print(f"  Fetched: {len(records):,} records")
    return records


# ── Flattening ─────────────────────────────────────────────────────────────────

def flatten(rec: dict) -> dict:
    def join_list(lst, key=None):
        if not lst:
            return ""
        if key:
            return "; ".join(str(x.get(key, "")) for x in lst if isinstance(x, dict))
        return "; ".join(str(x) for x in lst)

    authors_raw = rec.get("authors", []) or []
    authors_str = "; ".join(
        f"{a.get('last_name','')}, {a.get('first_name','')}".strip(", ")
        for a in authors_raw if isinstance(a, dict)
    )

    rpd = rec.get("resulting_publication_doi", []) or []
    rpd_str = "; ".join(rpd) if isinstance(rpd, list) else str(rpd)

    oa = rec.get("open_access", []) or []
    oa_str = "; ".join(str(o) for o in oa) if isinstance(oa, list) else str(oa)

    cats = rec.get("category_for", []) or []
    cat_str = join_list(cats, key="name")

    countries = rec.get("research_org_countries", []) or []
    country_str = join_list(countries, key="name")

    orgs = rec.get("research_org_names", []) or []
    orgs_str = "; ".join(str(o) for o in orgs)

    return {
        "dimensions_id":             rec.get("id", ""),
        "title":                     (rec.get("title", "") or "").replace("\n", " ").strip(),
        "abstract":                  (rec.get("abstract", "") or "").replace("\n", " ").strip(),
        "authors":                   authors_str,
        "year":                      rec.get("year", ""),
        "date":                      rec.get("date", ""),
        "doi":                       rec.get("doi", ""),
        "source_title":              rec.get("source_title", ""),
        "publisher":                 rec.get("publisher", ""),
        "resulting_publication_doi": rpd_str,
        "open_access":               oa_str,
        "field_of_research":         cat_str,
        "research_org_names":        orgs_str,
        "research_org_countries":    country_str,
    }


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    token = get_token(KEY_FILE)

    for server_key, cfg in SERVERS.items():
        output_path = cfg["output"]
        output_path.parent.mkdir(parents=True, exist_ok=True)

        records = fetch_all(cfg["label"], cfg["source_id"], token)

        if not records:
            print(f"  No records returned for {server_key}.")
            continue

        df = pd.DataFrame([flatten(r) for r in records])
        df.to_csv(output_path, index=False, encoding="utf-8")
        print(f"  Saved {len(df):,} records → {output_path}")

    print("\nStep 2 complete.")


if __name__ == "__main__":
    main()

"""
lookup_source_ids.py

Look up the Dimensions source title IDs for medRxiv and SSRN.
Run once to confirm IDs, then hardcode them in query_dimensions.py.

Usage:
    python3 code/collection/lookup_source_ids.py
"""

import requests
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[2]
KEY_FILE = BASE_DIR / "config" / "dimensions_key.txt"

key = KEY_FILE.read_text().strip()
token = requests.post(
    "https://app.dimensions.ai/api/auth.json",
    json={"key": key}, timeout=30
).json()["token"]

headers = {"Authorization": f"JWT {token}"}

for term in ["medRxiv", "SSRN"]:
    q = f'search source_titles\nfor "{term}"\nreturn source_titles[id+title]\nlimit 5 skip 0'
    r = requests.post("https://app.dimensions.ai/api/dsl.json", data=q, headers=headers, timeout=30)
    print(f"\n=== {term} ===")
    for st in r.json().get("source_titles", []):
        print(f"  id={st['id']}  title={st['title']}")

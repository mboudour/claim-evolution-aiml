"""
Filter the arXiv metadata snapshot for AI/ML categories, 2015-2024.

Target categories: cs.AI, cs.LG, cs.CL, cs.CV, cs.NE, stat.ML
Output: CSV with fields: arxiv_id, title, abstract, authors, categories,
        submitter, doi, journal_ref, update_date, first_submission_date
"""

import json
import csv
import os
import sys
from datetime import datetime

INPUT_FILE = "/home/ubuntu/data/raw/arxiv/snapshots/arxiv-metadata-oai-snapshot.json"
OUTPUT_DIR = "/home/ubuntu/data/raw/arxiv/processed"
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "arxiv_aiml_2015_2024.csv")

TARGET_CATEGORIES = {"cs.AI", "cs.LG", "cs.CL", "cs.CV", "cs.NE", "stat.ML"}
START_YEAR = 2015
END_YEAR = 2024

os.makedirs(OUTPUT_DIR, exist_ok=True)

def get_first_version_date(versions):
    """Extract the date of the first submitted version."""
    if versions and len(versions) > 0:
        created_str = versions[0].get("created", "")
        try:
            # Format: "Mon, 2 Apr 2007 19:18:42 GMT"
            dt = datetime.strptime(created_str, "%a, %d %b %Y %H:%M:%S %Z")
            return dt.strftime("%Y-%m-%d"), dt.year
        except Exception:
            pass
    return "", None

def paper_matches(record):
    """Return True if the paper belongs to target categories and year range."""
    cats = record.get("categories", "")
    cat_set = set(cats.split())
    if not cat_set.intersection(TARGET_CATEGORIES):
        return False
    _, year = get_first_version_date(record.get("versions", []))
    if year is None:
        return False
    return START_YEAR <= year <= END_YEAR

fieldnames = [
    "arxiv_id", "title", "abstract", "authors", "categories",
    "submitter", "doi", "journal_ref", "update_date", "first_submission_date",
    "first_submission_year"
]

total_read = 0
total_matched = 0
year_counts = {}

print(f"Reading {INPUT_FILE} ...")
print(f"Filtering for categories: {TARGET_CATEGORIES}")
print(f"Year range: {START_YEAR}–{END_YEAR}")
print()

with open(INPUT_FILE, "r", encoding="utf-8") as infile, \
     open(OUTPUT_FILE, "w", newline="", encoding="utf-8") as outfile:

    writer = csv.DictWriter(outfile, fieldnames=fieldnames)
    writer.writeheader()

    for line in infile:
        line = line.strip()
        if not line:
            continue
        total_read += 1

        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            continue

        if not paper_matches(record):
            continue

        first_date, first_year = get_first_version_date(record.get("versions", []))

        # Clean abstract: collapse whitespace
        abstract = record.get("abstract", "").replace("\n", " ").strip()
        title = record.get("title", "").replace("\n", " ").strip()

        # Authors: join parsed authors as "Last, First" strings
        authors_parsed = record.get("authors_parsed", [])
        if authors_parsed:
            authors_str = "; ".join(
                f"{a[0]}, {a[1]}".strip(", ")
                for a in authors_parsed
                if a
            )
        else:
            authors_str = record.get("authors", "")

        writer.writerow({
            "arxiv_id": record.get("id", ""),
            "title": title,
            "abstract": abstract,
            "authors": authors_str,
            "categories": record.get("categories", ""),
            "submitter": record.get("submitter", ""),
            "doi": record.get("doi", ""),
            "journal_ref": record.get("journal-ref", ""),
            "update_date": record.get("update_date", ""),
            "first_submission_date": first_date,
            "first_submission_year": first_year,
        })

        total_matched += 1
        year_counts[first_year] = year_counts.get(first_year, 0) + 1

        if total_read % 500_000 == 0:
            print(f"  Read {total_read:,} records, matched {total_matched:,} so far...")
            sys.stdout.flush()

print(f"\nDone.")
print(f"Total records read:    {total_read:,}")
print(f"Total records matched: {total_matched:,}")
print(f"\nBreakdown by year:")
for year in sorted(year_counts):
    print(f"  {year}: {year_counts[year]:,}")
print(f"\nOutput written to: {OUTPUT_FILE}")

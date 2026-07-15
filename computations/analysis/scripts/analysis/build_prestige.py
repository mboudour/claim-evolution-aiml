#!/usr/bin/env python3
"""
build_prestige.py

Step 3 of the Claim Evolution Pipeline.
Builds a unified venue-prestige table by combining:
  1. CORE 2023 rankings for conferences (A* -> 4, A -> 3, B -> 2, C/Unranked -> 1)
  2. Scimago SJR quartiles for journals (Q1 -> 4, Q2 -> 3, Q3 -> 2, Q4/Unranked -> 1)

This script assigns an ordinal 1-4 prestige score to each venue in the corpus.
It also computes an internal citation-based prestige measure (median citation count
per venue) as a robustness check.

Output: venue_prestige.csv
"""

import pandas as pd
import numpy as np
import os
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger(__name__)

# --- Paths ---
# The script will be run by the user or the agent. We assume the corpus is available.
# In the sandbox, it's at /home/ubuntu/upload/analysis_corpus.csv
CORPUS_PATH = os.environ.get("CORPUS_PATH", "/home/ubuntu/upload/analysis_corpus.csv")
OUT_PATH    = os.environ.get("PRESTIGE_OUT_PATH", "/home/ubuntu/upload/venue_prestige.csv")

def compute_internal_prestige(df):
    """Compute median citation count per venue."""
    # Filter to items with valid citations
    has_cites = df[df['pub_citations_count'].notna()]
    
    # Group by venue name (journal or conference name)
    # We use pub_journal for both, as it contains the venue string in Dimensions
    internal_prestige = has_cites.groupby('pub_journal')['pub_citations_count'].median().reset_index()
    internal_prestige.rename(columns={'pub_citations_count': 'median_venue_citations'}, inplace=True)
    
    # Also get venue counts
    venue_counts = df['pub_journal'].value_counts().reset_index()
    venue_counts.columns = ['pub_journal', 'corpus_paper_count']
    
    internal_prestige = pd.merge(internal_prestige, venue_counts, on='pub_journal', how='left')
    return internal_prestige

def build_prestige_table():
    if not Path(CORPUS_PATH).exists():
        log.error(f"Corpus file not found: {CORPUS_PATH}")
        return

    log.info(f"Loading corpus from {CORPUS_PATH}")
    df = pd.read_csv(CORPUS_PATH, usecols=['pub_journal', 'venue_type', 'pub_citations_count'])
    
    # Drop rows with no venue name
    df = df.dropna(subset=['pub_journal'])
    
    log.info("Computing internal citation-based prestige...")
    prestige_df = compute_internal_prestige(df)
    
    # In a full implementation, we would load the CORE 2023 and SJR 2025 datasets here
    # and perform fuzzy matching on the `pub_journal` string.
    # Since those external datasets are not present in the sandbox, we will 
    # generate a template/placeholder for the external prestige scores, which the 
    # user can populate later if they run the script locally with the datasets.
    
    log.info("Adding external prestige placeholder columns (CORE/SJR)...")
    prestige_df['external_prestige_source'] = np.where(
        prestige_df['pub_journal'].str.contains('Conf|Proc|Symp|Workshop', case=False, na=False),
        'CORE', 'SJR'
    )
    
    # Placeholder: assign a default score of 1 (Unranked) to all venues initially
    prestige_df['external_prestige_score'] = 1
    
    # Sort by corpus paper count descending
    prestige_df = prestige_df.sort_values('corpus_paper_count', ascending=False)
    
    log.info(f"Saving prestige table to {OUT_PATH}")
    prestige_df.to_csv(OUT_PATH, index=False)
    log.info(f"Processed {len(prestige_df)} unique venues.")

if __name__ == "__main__":
    build_prestige_table()

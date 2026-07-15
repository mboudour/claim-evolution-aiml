#!/usr/bin/env python3
"""
build_dataset.py
Step 4 of the Claim Evolution Pipeline.

Merges all upstream outputs into two flat analysis-ready CSV files:
  1. analysis_dataset.csv   — one row per claim alignment
  2. pair_level_dataset.csv — one row per paper pair (aggregated)

Inputs (all paths configurable via environment variables):
  CLAIMS_DIR        directory containing batch_NN_of_10.jsonl files
                    (or a single merged file at CLAIMS_JSONL)
  CORPUS_PATH       analysis_corpus.csv
  SUBFIELD_PATH     subfield_labels.csv
  PRESTIGE_PATH     venue_prestige.csv

Outputs:
  ALIGN_OUT_PATH    analysis_dataset.csv
  PAIR_OUT_PATH     pair_level_dataset.csv
"""

import json
import os
import glob
import numpy as np
import pandas as pd
from pathlib import Path

# ── Paths ─────────────────────────────────────────────────────────────────────
CLAIMS_DIR    = os.environ.get("CLAIMS_DIR",      "/home/ubuntu/upload")
CORPUS_PATH   = os.environ.get("CORPUS_PATH",     "/home/ubuntu/upload/analysis_corpus.csv")
SUBFIELD_PATH = os.environ.get("SUBFIELD_PATH",   "/home/ubuntu/upload/subfield_labels.csv")
PRESTIGE_PATH = os.environ.get("PRESTIGE_PATH",   "/home/ubuntu/upload/venue_prestige.csv")
ALIGN_OUT     = os.environ.get("ALIGN_OUT_PATH",  "/home/ubuntu/upload/analysis_dataset.csv")
PAIR_OUT      = os.environ.get("PAIR_OUT_PATH",   "/home/ubuntu/upload/pair_level_dataset.csv")

# ── Ordinal encodings ─────────────────────────────────────────────────────────
SEMANTIC_ORDER = {'Unchanged': 0, 'Clarified': 1, 'Revised': 2, 'Removed': 3, 'Added': 4}
SCOPE_ORDER    = {'Unchanged': 0, 'Narrowed': 1, 'Broadened': 2, 'N/A': np.nan}
CONF_ORDER     = {'Unchanged': 0, 'Tempered': 1, 'Amplified': 2, 'N/A': np.nan}


def load_annotations(claims_dir: str) -> pd.DataFrame:
    """Load all batch_NN_of_10.jsonl files and return a flat alignment DataFrame."""
    # Find all batch files
    batch_files = sorted(glob.glob(os.path.join(claims_dir, "batch_*_of_*.jsonl")))
    # Also accept a single merged file
    merged_file = os.path.join(claims_dir, "claim_changes_new.jsonl")
    if not batch_files and os.path.exists(merged_file):
        batch_files = [merged_file]

    if not batch_files:
        raise FileNotFoundError(
            f"No batch_NN_of_10.jsonl files found in {claims_dir} "
            f"and no claim_changes_new.jsonl either."
        )

    print(f"Loading annotations from {len(batch_files)} file(s)...")
    rows = []
    seen_pair_ids = set()

    for fpath in batch_files:
        with open(fpath) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)
                pair_id = str(rec.get('pair_id', ''))
                if pair_id in seen_pair_ids:
                    continue  # deduplicate across batches
                seen_pair_ids.add(pair_id)

                for aln in rec.get('alignments', []):
                    if not isinstance(aln, dict):
                        continue
                    rows.append({
                        'pair_id':             pair_id,
                        'preprint_claim_idx':  aln.get('preprint_claim_index', -1),
                        'pub_claim_idx':       aln.get('pub_claim_index', -1),
                        'preprint_claim_text': aln.get('preprint_claim_text', ''),
                        'pub_claim_text':      aln.get('pub_claim_text', ''),
                        'semantic':            aln.get('semantic', ''),
                        'scope':               aln.get('scope', ''),
                        'confidence':          aln.get('confidence', ''),
                        'matching_confidence': aln.get('matching_confidence', np.nan),
                        'rationale':           aln.get('rationale', ''),
                    })

    df = pd.DataFrame(rows)
    print(f"  {len(df):,} alignment rows from {len(seen_pair_ids):,} unique pairs")
    return df


def load_corpus_meta(corpus_path: str) -> pd.DataFrame:
    """Load and clean corpus metadata."""
    df = pd.read_csv(corpus_path, low_memory=False)
    if 'arxiv_id' in df.columns and 'pair_id' not in df.columns:
        df = df.rename(columns={'arxiv_id': 'pair_id'})
    df['pair_id'] = df['pair_id'].astype(str)

    meta_cols = [
        'pair_id', 'source', 'preprint_year', 'pub_year', 'pub_year_int',
        'venue_type', 'pub_venue', 'pub_journal', 'linkage_method',
        'pub_open_access', 'preprint_categories', 'field_of_research',
        'preprint_citation_count', 'pub_citation_count',
        'n_preprint_authors', 'n_pub_authors',
    ]
    meta_cols = [c for c in meta_cols if c in df.columns]
    meta = df[meta_cols].copy()

    # Version proxy covariate
    def _version_proxy(oa):
        if pd.isna(oa):
            return 'unknown'
        oa = str(oa).lower()
        if 'green' in oa:
            return 'green_OA'
        if oa in ('gold', 'hybrid', 'bronze', 'diamond'):
            return 'VoR_published'
        if oa == 'closed':
            return 'VoR_closed'
        return 'other'

    if 'pub_open_access' in meta.columns:
        meta['version_proxy'] = meta['pub_open_access'].apply(_version_proxy)

    print(f"  Corpus: {len(meta):,} rows")
    return meta


def main():
    # ── 1. Load annotations ───────────────────────────────────────────────────
    align_df = load_annotations(CLAIMS_DIR)

    # ── 2. Load metadata ──────────────────────────────────────────────────────
    print(f"Loading corpus from {CORPUS_PATH}")
    corpus = load_corpus_meta(CORPUS_PATH)

    print(f"Loading subfield labels from {SUBFIELD_PATH}")
    subfields = pd.read_csv(SUBFIELD_PATH)
    subfields['pair_id'] = subfields['pair_id'].astype(str)
    print(f"  {len(subfields):,} rows")

    prestige = None
    if os.path.exists(PRESTIGE_PATH):
        print(f"Loading venue prestige from {PRESTIGE_PATH}")
        prestige = pd.read_csv(PRESTIGE_PATH)
        print(f"  {len(prestige):,} rows")

    # ── 3. Build alignment-level dataset ─────────────────────────────────────
    print("Building alignment-level dataset...")
    df = align_df.merge(corpus, on='pair_id', how='left')
    df = df.merge(subfields, on='pair_id', how='left')

    if prestige is not None:
        venue_col = 'pub_venue' if 'pub_venue' in prestige.columns else prestige.columns[0]
        prestige_join = prestige[[venue_col, 'prestige_tier', 'n_papers']].copy() \
            if 'prestige_tier' in prestige.columns else prestige.iloc[:, :3].copy()
        prestige_join.columns = ['pub_venue', 'prestige_tier', 'n_papers']
        if 'pub_venue' in df.columns:
            df = df.merge(prestige_join, on='pub_venue', how='left')

    # Ordinal codes
    df['semantic_code']   = df['semantic'].map(SEMANTIC_ORDER)
    df['scope_code']      = df['scope'].map(SCOPE_ORDER)
    df['confidence_code'] = df['confidence'].map(CONF_ORDER)

    # Binary flags
    df['is_changed']   = (df['semantic'] != 'Unchanged').astype(int)
    df['is_tempered']  = (df['confidence'] == 'Tempered').astype(int)
    df['is_amplified'] = (df['confidence'] == 'Amplified').astype(int)
    df['is_narrowed']  = (df['scope'] == 'Narrowed').astype(int)
    df['is_broadened'] = (df['scope'] == 'Broadened').astype(int)
    df['is_removed']   = (df['semantic'] == 'Removed').astype(int)
    df['is_added']     = (df['semantic'] == 'Added').astype(int)

    df.to_csv(ALIGN_OUT, index=False)
    print(f"  Saved: {ALIGN_OUT} ({len(df):,} rows, {len(df.columns)} cols)")

    # ── 4. Build pair-level dataset ───────────────────────────────────────────
    print("Building pair-level dataset...")
    agg = df.groupby('pair_id').agg(
        n_alignments  = ('semantic', 'count'),
        n_unchanged   = ('semantic', lambda x: (x == 'Unchanged').sum()),
        n_clarified   = ('semantic', lambda x: (x == 'Clarified').sum()),
        n_revised     = ('semantic', lambda x: (x == 'Revised').sum()),
        n_removed     = ('semantic', lambda x: (x == 'Removed').sum()),
        n_added       = ('semantic', lambda x: (x == 'Added').sum()),
        n_tempered    = ('confidence', lambda x: (x == 'Tempered').sum()),
        n_amplified   = ('confidence', lambda x: (x == 'Amplified').sum()),
        n_narrowed    = ('scope', lambda x: (x == 'Narrowed').sum()),
        n_broadened   = ('scope', lambda x: (x == 'Broadened').sum()),
        mean_match_conf = ('matching_confidence', 'mean'),
    ).reset_index()

    # Proportions
    for col in ['unchanged', 'clarified', 'revised', 'removed', 'added',
                'tempered', 'amplified', 'narrowed', 'broadened']:
        agg[f'pct_{col}'] = agg[f'n_{col}'] / agg['n_alignments']
    agg['pct_changed'] = 1 - agg['pct_unchanged']

    # Merge metadata back
    meta_for_pair = corpus.drop_duplicates('pair_id')
    agg = agg.merge(meta_for_pair, on='pair_id', how='left')
    agg = agg.merge(subfields, on='pair_id', how='left')
    if prestige is not None and 'pub_venue' in agg.columns:
        agg = agg.merge(prestige_join, on='pub_venue', how='left')

    agg.to_csv(PAIR_OUT, index=False)
    print(f"  Saved: {PAIR_OUT} ({len(agg):,} rows, {len(agg.columns)} cols)")

    # ── 5. Summary ────────────────────────────────────────────────────────────
    print("\n=== DATASET SUMMARY ===")
    print(f"Total paper pairs:      {agg['pair_id'].nunique():,}")
    print(f"Total claim alignments: {len(df):,}")
    print(f"\nSemantic change distribution:")
    print(df['semantic'].value_counts().to_string())
    print(f"\nScope change distribution:")
    print(df['scope'].value_counts().to_string())
    print(f"\nConfidence change distribution:")
    print(df['confidence'].value_counts().to_string())
    if 'subfield' in agg.columns:
        print(f"\nSubfield distribution (pairs):")
        print(agg['subfield'].value_counts().to_string())
    print("\n=== DONE ===")


if __name__ == '__main__':
    main()

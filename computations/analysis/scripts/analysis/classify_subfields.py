#!/usr/bin/env python3
"""
classify_subfields.py
Step 2 of the Claim Evolution Pipeline.

Maps arXiv primary categories to a set of 13 AI/ML subfield labels using a
deterministic lookup table. No LLM calls required.

Input:  analysis_corpus.csv  (must contain 'arxiv_id' and 'preprint_categories')
Output: subfield_labels.csv  (pair_id, subfield)

Subfield taxonomy:
  Machine Learning, Computer Vision, NLP, AI & Knowledge,
  Robotics, Signal Processing, Biomedical AI, HCI,
  Speech & Audio, Systems, Theory, Multimodal, Other
"""

import os
import pandas as pd

CORPUS_PATH  = os.environ.get("CORPUS_PATH",    "/home/ubuntu/upload/analysis_corpus.csv")
OUT_PATH     = os.environ.get("SUBFIELD_OUT_PATH", "/home/ubuntu/upload/subfield_labels.csv")

# Priority-ordered mapping: first matching category wins
CATEGORY_MAP = {
    # NLP / Information Retrieval
    'cs.CL':   'NLP',
    'cs.IR':   'NLP',
    # Computer Vision
    'cs.CV':   'Computer Vision',
    'eess.IV': 'Computer Vision',
    # Machine Learning (general)
    'cs.LG':   'Machine Learning',
    'stat.ML': 'Machine Learning',
    # AI / Knowledge Representation / Planning
    'cs.AI':   'AI & Knowledge',
    'cs.KR':   'AI & Knowledge',
    # Robotics
    'cs.RO':   'Robotics',
    # Human-Computer Interaction
    'cs.HC':   'HCI',
    # Systems / Architecture / Distributed
    'cs.AR':   'Systems',
    'cs.DC':   'Systems',
    'cs.OS':   'Systems',
    'cs.PF':   'Systems',
    # Theory / Logic / Complexity
    'cs.LO':   'Theory',
    'cs.CC':   'Theory',
    'math.OC': 'Theory',
    'stat.TH': 'Theory',
    # Bioinformatics / Medical AI
    'q-bio':   'Biomedical AI',   # prefix match
    'cs.CE':   'Biomedical AI',
    # Speech / Audio
    'cs.SD':   'Speech & Audio',
    'eess.AS': 'Speech & Audio',
    # Signal Processing
    'eess.SP': 'Signal Processing',
    'eess.SY': 'Signal Processing',
    # Multimodal / Graphics
    'cs.MM':   'Multimodal',
    'cs.GR':   'Multimodal',
}


def assign_subfield(cats_str: str) -> str:
    """Return the first matching subfield for a space/semicolon-separated category string."""
    if pd.isna(cats_str) or not cats_str:
        return 'Other'
    cats = [c.strip() for c in str(cats_str).replace(';', ' ').replace(',', ' ').split()]
    for cat in cats:
        if cat in CATEGORY_MAP:
            return CATEGORY_MAP[cat]
        # Prefix match (e.g. q-bio.QM → Biomedical AI)
        for prefix, subfield in CATEGORY_MAP.items():
            if cat.startswith(prefix):
                return subfield
    return 'Other'


def main():
    print(f"Loading corpus from {CORPUS_PATH}")
    df = pd.read_csv(CORPUS_PATH, low_memory=False)
    print(f"  {len(df):,} rows loaded")

    # Use arxiv_id as join key (= pair_id in annotation output)
    id_col = 'arxiv_id' if 'arxiv_id' in df.columns else 'pair_id'
    df['subfield'] = df['preprint_categories'].apply(assign_subfield)

    print("\nSubfield distribution:")
    print(df['subfield'].value_counts().to_string())

    out = df[[id_col, 'subfield']].rename(columns={id_col: 'pair_id'})
    out.to_csv(OUT_PATH, index=False)
    print(f"\nSaved: {OUT_PATH} ({len(out):,} rows)")


if __name__ == '__main__':
    main()

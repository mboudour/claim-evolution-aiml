#!/usr/bin/env python3
"""
make_figures.py
Step 7 of the Claim Evolution Pipeline.

Produces all publication-quality figures for the manuscript:

  Fig 1. Overall distribution of semantic change labels (stacked bar)
  Fig 2. Scope change distribution (Narrowed / Unchanged / Broadened)
  Fig 3. Confidence change distribution (Tempered / Unchanged / Amplified)
  Fig 4. Change rates by AI/ML subfield (grouped bar)
  Fig 5. Temporal trends: pct_changed by publication year
  Fig 6. Prestige tier vs. pct_tempered (box plot)
  Fig 7. Version proxy vs. pct_changed (box plot)
  Fig 8. Heatmap: semantic × confidence co-occurrence

Input:  pair_level_dataset.csv, analysis_dataset.csv
Output: /home/ubuntu/upload/figures/*.pdf  (and .png for quick preview)
"""

import os
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import seaborn as sns

warnings.filterwarnings('ignore')

PAIR_CSV    = os.environ.get("PAIR_OUT_PATH",  "/home/ubuntu/upload/pair_level_dataset.csv")
ALIGN_CSV   = os.environ.get("ALIGN_OUT_PATH", "/home/ubuntu/upload/analysis_dataset.csv")
FIG_DIR     = os.environ.get("FIG_DIR",        "/home/ubuntu/upload/figures")

os.makedirs(FIG_DIR, exist_ok=True)

# ── Style ─────────────────────────────────────────────────────────────────────
PALETTE = {
    'Unchanged':  '#4e79a7',
    'Clarified':  '#59a14f',
    'Revised':    '#f28e2b',
    'Removed':    '#e15759',
    'Added':      '#76b7b2',
    'Tempered':   '#59a14f',
    'Amplified':  '#e15759',
    'Narrowed':   '#59a14f',
    'Broadened':  '#e15759',
    'N/A':        '#bab0ac',
}

plt.rcParams.update({
    'font.family':    'DejaVu Sans',
    'font.size':      11,
    'axes.titlesize': 13,
    'axes.labelsize': 12,
    'figure.dpi':     150,
})


def save(fig, name):
    for ext in ('pdf', 'png'):
        path = os.path.join(FIG_DIR, f"{name}.{ext}")
        fig.savefig(path, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {name}.pdf / .png")


def fig1_semantic_distribution(align_df):
    """Overall semantic change label distribution."""
    counts = align_df['semantic'].value_counts().reindex(
        ['Unchanged', 'Clarified', 'Revised', 'Removed', 'Added'], fill_value=0
    )
    pcts = counts / counts.sum() * 100

    fig, ax = plt.subplots(figsize=(7, 4))
    bars = ax.bar(pcts.index, pcts.values,
                  color=[PALETTE[l] for l in pcts.index], edgecolor='white', linewidth=0.5)
    for bar, pct in zip(bars, pcts.values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.3,
                f'{pct:.1f}%', ha='center', va='bottom', fontsize=10)
    ax.set_ylabel('Percentage of claim alignments (%)')
    ax.set_title('Fig 1. Distribution of Semantic Change Labels')
    ax.yaxis.set_major_formatter(mtick.PercentFormatter())
    ax.set_ylim(0, pcts.max() * 1.15)
    fig.tight_layout()
    save(fig, 'fig1_semantic_distribution')


def fig2_scope_distribution(align_df):
    """Scope change distribution."""
    counts = align_df['scope'].value_counts().reindex(
        ['Unchanged', 'Narrowed', 'Broadened', 'N/A'], fill_value=0
    )
    pcts = counts / counts.sum() * 100

    fig, ax = plt.subplots(figsize=(6, 4))
    bars = ax.bar(pcts.index, pcts.values,
                  color=[PALETTE.get(l, '#bab0ac') for l in pcts.index],
                  edgecolor='white', linewidth=0.5)
    for bar, pct in zip(bars, pcts.values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.3,
                f'{pct:.1f}%', ha='center', va='bottom', fontsize=10)
    ax.set_ylabel('Percentage of claim alignments (%)')
    ax.set_title('Fig 2. Distribution of Scope Change Labels')
    ax.yaxis.set_major_formatter(mtick.PercentFormatter())
    ax.set_ylim(0, pcts.max() * 1.15)
    fig.tight_layout()
    save(fig, 'fig2_scope_distribution')


def fig3_confidence_distribution(align_df):
    """Confidence change distribution."""
    counts = align_df['confidence'].value_counts().reindex(
        ['Unchanged', 'Tempered', 'Amplified', 'N/A'], fill_value=0
    )
    pcts = counts / counts.sum() * 100

    fig, ax = plt.subplots(figsize=(6, 4))
    bars = ax.bar(pcts.index, pcts.values,
                  color=[PALETTE.get(l, '#bab0ac') for l in pcts.index],
                  edgecolor='white', linewidth=0.5)
    for bar, pct in zip(bars, pcts.values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.3,
                f'{pct:.1f}%', ha='center', va='bottom', fontsize=10)
    ax.set_ylabel('Percentage of claim alignments (%)')
    ax.set_title('Fig 3. Distribution of Confidence Change Labels')
    ax.yaxis.set_major_formatter(mtick.PercentFormatter())
    ax.set_ylim(0, pcts.max() * 1.15)
    fig.tight_layout()
    save(fig, 'fig3_confidence_distribution')


def fig4_change_by_subfield(pair_df):
    """Change rates by AI/ML subfield."""
    if 'subfield' not in pair_df.columns or 'pct_changed' not in pair_df.columns:
        print("  Skipping Fig 4: missing columns")
        return

    sf_stats = pair_df.groupby('subfield').agg(
        pct_changed  = ('pct_changed',  'mean'),
        pct_tempered = ('pct_tempered', 'mean'),
        pct_amplified= ('pct_amplified','mean'),
        n            = ('pair_id', 'count'),
    ).reset_index()
    sf_stats = sf_stats[sf_stats['n'] >= 30].sort_values('pct_changed', ascending=False)

    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(len(sf_stats))
    w = 0.28
    ax.bar(x - w,   sf_stats['pct_changed']   * 100, w, label='Any change',  color='#4e79a7')
    ax.bar(x,       sf_stats['pct_tempered']  * 100, w, label='Tempered',    color='#59a14f')
    ax.bar(x + w,   sf_stats['pct_amplified'] * 100, w, label='Amplified',   color='#e15759')
    ax.set_xticks(x)
    ax.set_xticklabels(sf_stats['subfield'], rotation=30, ha='right')
    ax.set_ylabel('Mean % of claims per pair (%)')
    ax.set_title('Fig 4. Claim Change Rates by AI/ML Subfield')
    ax.legend()
    ax.yaxis.set_major_formatter(mtick.PercentFormatter())
    fig.tight_layout()
    save(fig, 'fig4_change_by_subfield')


def fig5_temporal_trends(pair_df):
    """Temporal trends in pct_changed by publication year."""
    year_col = 'pub_year_int' if 'pub_year_int' in pair_df.columns else 'pub_year'
    if year_col not in pair_df.columns or 'pct_changed' not in pair_df.columns:
        print("  Skipping Fig 5: missing columns")
        return

    df = pair_df.copy()
    df['year'] = pd.to_numeric(df[year_col], errors='coerce')
    df = df[(df['year'] >= 2014) & (df['year'] <= 2025)]

    yearly = df.groupby('year').agg(
        pct_changed  = ('pct_changed',  'mean'),
        pct_tempered = ('pct_tempered', 'mean'),
        pct_amplified= ('pct_amplified','mean'),
        n            = ('pair_id', 'count'),
    ).reset_index()
    yearly = yearly[yearly['n'] >= 20]

    fig, ax = plt.subplots(figsize=(9, 4))
    ax.plot(yearly['year'], yearly['pct_changed']   * 100, 'o-', color='#4e79a7', label='Any change')
    ax.plot(yearly['year'], yearly['pct_tempered']  * 100, 's--', color='#59a14f', label='Tempered')
    ax.plot(yearly['year'], yearly['pct_amplified'] * 100, '^--', color='#e15759', label='Amplified')
    ax.set_xlabel('Publication year')
    ax.set_ylabel('Mean % of claims per pair (%)')
    ax.set_title('Fig 5. Temporal Trends in Claim Change Rates (2014–2025)')
    ax.legend()
    ax.yaxis.set_major_formatter(mtick.PercentFormatter())
    ax.set_xticks(yearly['year'].astype(int))
    ax.tick_params(axis='x', rotation=45)
    fig.tight_layout()
    save(fig, 'fig5_temporal_trends')


def fig6_prestige_vs_tempered(pair_df):
    """Prestige tier vs. pct_tempered box plot."""
    if 'prestige_tier' not in pair_df.columns or 'pct_tempered' not in pair_df.columns:
        print("  Skipping Fig 6: missing columns")
        return

    df = pair_df[pair_df['prestige_tier'].notna()].copy()
    df['prestige_tier'] = pd.to_numeric(df['prestige_tier'], errors='coerce')
    df = df[df['prestige_tier'].notna()]
    df['Prestige tier'] = df['prestige_tier'].astype(int).map(
        {1: '1 (Low)', 2: '2', 3: '3', 4: '4 (High)'}
    )

    fig, ax = plt.subplots(figsize=(7, 4))
    sns.boxplot(data=df, x='Prestige tier', y='pct_tempered',
                order=['1 (Low)', '2', '3', '4 (High)'],
                palette='Greens', ax=ax)
    ax.set_ylabel('Proportion of tempered claims')
    ax.set_title('Fig 6. Venue Prestige vs. Claim Tempering Rate')
    ax.yaxis.set_major_formatter(mtick.PercentFormatter(1.0))
    fig.tight_layout()
    save(fig, 'fig6_prestige_vs_tempered')


def fig7_version_proxy_vs_change(pair_df):
    """Version proxy vs. pct_changed box plot."""
    if 'version_proxy' not in pair_df.columns or 'pct_changed' not in pair_df.columns:
        print("  Skipping Fig 7: missing columns")
        return

    order = ['VoR_published', 'VoR_closed', 'green_OA', 'other', 'unknown']
    order = [o for o in order if o in pair_df['version_proxy'].unique()]

    fig, ax = plt.subplots(figsize=(7, 4))
    sns.boxplot(data=pair_df, x='version_proxy', y='pct_changed',
                order=order, palette='Blues', ax=ax)
    ax.set_xlabel('Publication version proxy')
    ax.set_ylabel('Proportion of changed claims')
    ax.set_title('Fig 7. Version Proxy vs. Overall Claim Change Rate')
    ax.yaxis.set_major_formatter(mtick.PercentFormatter(1.0))
    ax.tick_params(axis='x', rotation=15)
    fig.tight_layout()
    save(fig, 'fig7_version_proxy_vs_change')


def fig8_semantic_confidence_heatmap(align_df):
    """Heatmap of semantic × confidence co-occurrence."""
    if 'semantic' not in align_df.columns or 'confidence' not in align_df.columns:
        print("  Skipping Fig 8: missing columns")
        return

    ct = pd.crosstab(align_df['semantic'], align_df['confidence'], normalize='index') * 100
    ct = ct.reindex(index=['Unchanged', 'Clarified', 'Revised', 'Removed', 'Added'],
                    columns=['Unchanged', 'Tempered', 'Amplified', 'N/A'],
                    fill_value=0)

    fig, ax = plt.subplots(figsize=(7, 4))
    sns.heatmap(ct, annot=True, fmt='.1f', cmap='YlOrRd', ax=ax,
                cbar_kws={'label': '% of row'}, linewidths=0.5)
    ax.set_xlabel('Confidence change')
    ax.set_ylabel('Semantic change')
    ax.set_title('Fig 8. Semantic × Confidence Co-occurrence (%)')
    fig.tight_layout()
    save(fig, 'fig8_semantic_confidence_heatmap')


def main():
    print("Loading datasets...")
    pair_df  = pd.read_csv(PAIR_CSV,  low_memory=False) if os.path.exists(PAIR_CSV)  else pd.DataFrame()
    align_df = pd.read_csv(ALIGN_CSV, low_memory=False) if os.path.exists(ALIGN_CSV) else pd.DataFrame()

    if pair_df.empty and align_df.empty:
        print("No data found. Run build_dataset.py first.")
        return

    print(f"  Pair-level: {len(pair_df):,} rows")
    print(f"  Alignment-level: {len(align_df):,} rows")
    print(f"\nGenerating figures in {FIG_DIR}/...")

    if not align_df.empty:
        fig1_semantic_distribution(align_df)
        fig2_scope_distribution(align_df)
        fig3_confidence_distribution(align_df)
        fig8_semantic_confidence_heatmap(align_df)

    if not pair_df.empty:
        fig4_change_by_subfield(pair_df)
        fig5_temporal_trends(pair_df)
        fig6_prestige_vs_tempered(pair_df)
        fig7_version_proxy_vs_change(pair_df)

    print("\n=== ALL FIGURES SAVED ===")


if __name__ == '__main__':
    main()

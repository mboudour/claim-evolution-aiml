"""
statistical_analysis.py  —  Step 10 (Statistical Analysis & Regression)

Produces all quantitative results for the paper:

  1. Descriptive statistics — change type rates by source, venue, year, field
  2. Logistic regressions — predictors of claim strengthening vs weakening
     (at both claim level and pair level)
  3. Temporal trends — change rates by preprint year
  4. Venue effects — journal article vs conference paper vs book chapter
  5. Source effects — arXiv vs bioRxiv vs medRxiv
  6. Time-to-publication effect — same year vs 1 year vs 2+ years
  7. Publication figures — bar charts, trend lines, heatmaps

Reads from:
    data/claims/claim_changes_flat.csv   ← claim-level data
    data/claims/claim_changes.jsonl      ← pair-level data
    data/final/analysis_corpus.csv       ← corpus metadata (for covariates)

Writes to:
    data/analysis/regression_results.csv
    data/analysis/descriptive_stats.csv
    data/analysis/figures/              ← PNG figures

Usage:
    cd SSRN_bioRxiv_medRxiv_data_collection_via_Dimensions
    pip install pandas numpy scipy statsmodels matplotlib seaborn
    python3 code/analysis/statistical_analysis.py

Requirements:
    pip install pandas numpy scipy statsmodels matplotlib seaborn
"""

import json
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from scipy import stats
from statsmodels.formula.api import logit, mnlogit
from statsmodels.tools.sm_exceptions import ConvergenceWarning

warnings.filterwarnings("ignore", category=ConvergenceWarning)

# ── Paths ──────────────────────────────────────────────────────────────────────
SCRIPT_DIR   = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[1]

FLAT_FILE    = PROJECT_ROOT / "data" / "claims" / "claim_changes_flat.csv"
CHANGES_FILE = PROJECT_ROOT / "data" / "claims" / "claim_changes.jsonl"
CORPUS_FILE  = PROJECT_ROOT / "data" / "final" / "analysis_corpus.csv"

OUT_DIR      = PROJECT_ROOT / "data" / "analysis"
FIG_DIR      = OUT_DIR / "figures"
OUT_STATS    = OUT_DIR / "descriptive_stats.csv"
OUT_REGR     = OUT_DIR / "regression_results.csv"
OUT_REPORT   = OUT_DIR / "analysis_report.txt"

OUT_DIR.mkdir(parents=True, exist_ok=True)
FIG_DIR.mkdir(parents=True, exist_ok=True)

# ── Colour palette ─────────────────────────────────────────────────────────────
CHANGE_COLOURS = {
    "strengthened": "#2ecc71",
    "weakened":     "#e74c3c",
    "unchanged":    "#95a5a6",
    "removed":      "#e67e22",
    "added":        "#3498db",
    "mixed":        "#9b59b6",
}

# ── Load data ──────────────────────────────────────────────────────────────────

def load_data():
    print("Loading data...")

    # Claim-level flat CSV
    flat = pd.read_csv(FLAT_FILE, dtype=str, low_memory=False)
    flat["preprint_year"] = pd.to_numeric(flat["preprint_year"], errors="coerce")
    print(f"  Claim comparisons : {len(flat):,}")

    # Pair-level JSONL
    pair_records = []
    with open(CHANGES_FILE, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    rec = json.loads(line)
                    if not rec.get("error"):
                        pair_records.append(rec)
                except json.JSONDecodeError:
                    pass
    pairs = pd.DataFrame(pair_records)
    pairs["preprint_year"] = pd.to_numeric(pairs["preprint_year"], errors="coerce")
    pairs["pub_year"]      = pd.to_numeric(pairs["pub_year"],      errors="coerce")
    pairs["years_to_pub"]  = pairs["pub_year"] - pairs["preprint_year"]
    print(f"  Pairs             : {len(pairs):,}")

    # Corpus metadata (for additional covariates)
    corpus = pd.read_csv(CORPUS_FILE, dtype=str, low_memory=False)
    corpus["preprint_year"] = pd.to_numeric(corpus["preprint_year"], errors="coerce")
    corpus["years_to_pub"]  = pd.to_numeric(corpus["years_to_pub"],  errors="coerce")
    corpus["pub_citations_count"] = pd.to_numeric(
        corpus["pub_citations_count"], errors="coerce"
    )
    print(f"  Corpus            : {len(corpus):,}\n")

    # Merge years_to_pub and citations onto pairs
    corpus_slim = corpus[["doi", "pub_doi", "years_to_pub",
                           "pub_citations_count", "is_open_access"]].copy()
    corpus_slim = corpus_slim.rename(columns={"doi": "preprint_doi_corpus"})
    pairs = pairs.merge(
        corpus_slim.drop_duplicates("pub_doi"),
        on="pub_doi", how="left"
    )

    return flat, pairs, corpus

# ── 1. Descriptive statistics ──────────────────────────────────────────────────

def descriptive_stats(flat: pd.DataFrame, pairs: pd.DataFrame) -> pd.DataFrame:
    print("── 1. Descriptive statistics ──")

    valid_changes = ["strengthened", "weakened", "unchanged", "removed", "added"]
    flat_v = flat[flat["change_type"].isin(valid_changes)].copy()

    rows = []

    def add_row(group_name, group_val, subset):
        total = len(subset)
        if total == 0:
            return
        for ct in valid_changes:
            n = (subset["change_type"] == ct).sum()
            rows.append({
                "group":      group_name,
                "value":      group_val,
                "change_type": ct,
                "n":          n,
                "total":      total,
                "pct":        round(n / total * 100, 2),
            })

    # Overall
    add_row("overall", "all", flat_v)

    # By source
    for src in flat_v["source"].dropna().unique():
        add_row("source", src, flat_v[flat_v["source"] == src])

    # By venue type
    for vt in flat_v["venue_type"].dropna().unique():
        add_row("venue_type", vt, flat_v[flat_v["venue_type"] == vt])

    # By preprint year
    for yr in sorted(flat_v["preprint_year"].dropna().unique()):
        add_row("preprint_year", int(yr), flat_v[flat_v["preprint_year"] == yr])

    stats_df = pd.DataFrame(rows)
    stats_df.to_csv(OUT_STATS, index=False, encoding="utf-8")
    print(f"  Saved → {OUT_STATS.name}\n")
    return stats_df

# ── 2. Logistic regressions ────────────────────────────────────────────────────

def run_regressions(flat: pd.DataFrame, pairs: pd.DataFrame) -> pd.DataFrame:
    print("── 2. Logistic regressions ──")

    reg_rows = []

    # ── 2a. Claim-level: strengthened vs weakened ──────────────────────────────
    # Restrict to directional changes only
    dir_flat = flat[flat["change_type"].isin(["strengthened", "weakened"])].copy()
    dir_flat["y"] = (dir_flat["change_type"] == "strengthened").astype(int)
    dir_flat["preprint_year_c"] = dir_flat["preprint_year"] - 2019  # centre on 2019

    # Dummies
    dir_flat["is_arxiv"]   = (dir_flat["source"] == "arxiv").astype(int)
    dir_flat["is_journal"] = (dir_flat["venue_type"] == "journal_article").astype(int)
    dir_flat["is_conf"]    = (dir_flat["venue_type"] == "conference_paper").astype(int)

    try:
        m1 = logit(
            "y ~ preprint_year_c + is_arxiv + is_journal + is_conf",
            data=dir_flat.dropna(subset=["preprint_year_c"])
        ).fit(disp=0)

        for name, coef, se, pval in zip(
            m1.params.index, m1.params, m1.bse, m1.pvalues
        ):
            reg_rows.append({
                "model":     "claim_level_strengthened_vs_weakened",
                "predictor": name,
                "coef":      round(coef, 4),
                "se":        round(se, 4),
                "pvalue":    round(pval, 4),
                "OR":        round(np.exp(coef), 4),
                "n":         int(m1.nobs),
            })
        print(f"  Claim-level model: n={int(m1.nobs):,}, "
              f"pseudo-R²={m1.prsquared:.4f}")
    except Exception as e:
        print(f"  Claim-level model failed: {e}")

    # ── 2b. Pair-level: dominant weakened vs dominant strengthened ─────────────
    dir_pairs = pairs[pairs["dominant_change"].isin(["strengthened", "weakened"])].copy()
    dir_pairs["y"] = (dir_pairs["dominant_change"] == "strengthened").astype(int)
    dir_pairs["preprint_year_c"] = dir_pairs["preprint_year"] - 2019
    dir_pairs["is_arxiv"]   = (dir_pairs["source"] == "arxiv").astype(int)
    dir_pairs["is_journal"] = (dir_pairs["venue_type"] == "journal_article").astype(int)
    dir_pairs["is_conf"]    = (dir_pairs["venue_type"] == "conference_paper").astype(int)

    # Merge years_to_pub
    if "years_to_pub_x" in dir_pairs.columns:
        dir_pairs["years_to_pub"] = dir_pairs["years_to_pub_x"].fillna(
            dir_pairs.get("years_to_pub_y", np.nan)
        )

    try:
        m2 = logit(
            "y ~ preprint_year_c + is_arxiv + is_journal + is_conf + years_to_pub",
            data=dir_pairs.dropna(subset=["preprint_year_c", "years_to_pub"])
        ).fit(disp=0)

        for name, coef, se, pval in zip(
            m2.params.index, m2.params, m2.bse, m2.pvalues
        ):
            reg_rows.append({
                "model":     "pair_level_strengthened_vs_weakened",
                "predictor": name,
                "coef":      round(coef, 4),
                "se":        round(se, 4),
                "pvalue":    round(pval, 4),
                "OR":        round(np.exp(coef), 4),
                "n":         int(m2.nobs),
            })
        print(f"  Pair-level model:  n={int(m2.nobs):,}, "
              f"pseudo-R²={m2.prsquared:.4f}")
    except Exception as e:
        print(f"  Pair-level model failed: {e}")

    reg_df = pd.DataFrame(reg_rows)
    reg_df.to_csv(OUT_REGR, index=False, encoding="utf-8")
    print(f"  Saved → {OUT_REGR.name}\n")
    return reg_df

# ── 3. Figures ─────────────────────────────────────────────────────────────────

def make_figures(flat: pd.DataFrame, pairs: pd.DataFrame):
    print("── 3. Generating figures ──")
    valid = ["strengthened", "weakened", "unchanged", "removed", "added"]
    flat_v = flat[flat["change_type"].isin(valid)].copy()

    sns.set_theme(style="whitegrid", font_scale=1.1)

    # ── Fig 1: Overall change type distribution ────────────────────────────────
    fig, ax = plt.subplots(figsize=(8, 5))
    ct_counts = flat_v["change_type"].value_counts().reindex(valid).fillna(0)
    ct_pct    = ct_counts / ct_counts.sum() * 100
    bars = ax.bar(ct_pct.index, ct_pct.values,
                  color=[CHANGE_COLOURS[c] for c in ct_pct.index])
    ax.set_xlabel("Change Type")
    ax.set_ylabel("Percentage of Claim Comparisons (%)")
    ax.set_title("Distribution of Claim Changes\n(Preprint → Publication)")
    for bar, pct in zip(bars, ct_pct.values):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
                f"{pct:.1f}%", ha="center", va="bottom", fontsize=10)
    plt.tight_layout()
    fig.savefig(FIG_DIR / "fig1_overall_change_distribution.png", dpi=150)
    plt.close()
    print("  Fig 1: overall distribution")

    # ── Fig 2: Change rates by source ─────────────────────────────────────────
    src_pivot = (
        flat_v.groupby(["source", "change_type"])
        .size().unstack(fill_value=0)
        .reindex(columns=valid, fill_value=0)
    )
    src_pct = src_pivot.div(src_pivot.sum(axis=1), axis=0) * 100

    fig, ax = plt.subplots(figsize=(9, 5))
    src_pct.plot(kind="bar", ax=ax,
                 color=[CHANGE_COLOURS[c] for c in valid])
    ax.set_xlabel("Preprint Source")
    ax.set_ylabel("Percentage (%)")
    ax.set_title("Claim Change Rates by Preprint Source")
    ax.legend(title="Change Type", bbox_to_anchor=(1.01, 1), loc="upper left")
    plt.xticks(rotation=0)
    plt.tight_layout()
    fig.savefig(FIG_DIR / "fig2_change_by_source.png", dpi=150)
    plt.close()
    print("  Fig 2: by source")

    # ── Fig 3: Temporal trend (strengthened vs weakened) ──────────────────────
    yr_pivot = (
        flat_v[flat_v["change_type"].isin(["strengthened", "weakened"])]
        .groupby(["preprint_year", "change_type"])
        .size().unstack(fill_value=0)
    )
    yr_total = flat_v.groupby("preprint_year")["change_type"].count()
    yr_pct   = yr_pivot.div(yr_total, axis=0) * 100

    fig, ax = plt.subplots(figsize=(10, 5))
    for ct in ["strengthened", "weakened"]:
        if ct in yr_pct.columns:
            ax.plot(yr_pct.index.astype(int), yr_pct[ct],
                    marker="o", label=ct.capitalize(),
                    color=CHANGE_COLOURS[ct], linewidth=2)
    ax.set_xlabel("Preprint Year")
    ax.set_ylabel("% of All Claim Comparisons")
    ax.set_title("Temporal Trend: Strengthened vs Weakened Claims")
    ax.legend()
    ax.set_xticks(yr_pct.index.astype(int))
    plt.tight_layout()
    fig.savefig(FIG_DIR / "fig3_temporal_trend.png", dpi=150)
    plt.close()
    print("  Fig 3: temporal trend")

    # ── Fig 4: Change rates by venue type ─────────────────────────────────────
    vt_order = ["journal_article", "conference_paper", "book_chapter"]
    vt_pivot = (
        flat_v[flat_v["venue_type"].isin(vt_order)]
        .groupby(["venue_type", "change_type"])
        .size().unstack(fill_value=0)
        .reindex(index=vt_order, columns=valid, fill_value=0)
    )
    vt_pct = vt_pivot.div(vt_pivot.sum(axis=1), axis=0) * 100

    fig, ax = plt.subplots(figsize=(9, 5))
    vt_pct.plot(kind="bar", ax=ax,
                color=[CHANGE_COLOURS[c] for c in valid])
    ax.set_xlabel("Venue Type")
    ax.set_ylabel("Percentage (%)")
    ax.set_title("Claim Change Rates by Publication Venue Type")
    ax.legend(title="Change Type", bbox_to_anchor=(1.01, 1), loc="upper left")
    plt.xticks(rotation=15, ha="right")
    plt.tight_layout()
    fig.savefig(FIG_DIR / "fig4_change_by_venue.png", dpi=150)
    plt.close()
    print("  Fig 4: by venue type")

    # ── Fig 5: Dominant change heatmap (source × venue) ───────────────────────
    dom_valid = ["strengthened", "weakened", "unchanged", "mixed"]
    pairs_v   = pairs[pairs["dominant_change"].isin(dom_valid)].copy()
    hm_data   = (
        pairs_v.groupby(["source", "dominant_change"])
        .size().unstack(fill_value=0)
        .reindex(columns=dom_valid, fill_value=0)
    )
    hm_pct = hm_data.div(hm_data.sum(axis=1), axis=0) * 100

    fig, ax = plt.subplots(figsize=(8, 4))
    sns.heatmap(hm_pct, annot=True, fmt=".1f", cmap="RdYlGn",
                linewidths=0.5, ax=ax, cbar_kws={"label": "%"})
    ax.set_title("Dominant Change Type by Source (% of pairs)")
    ax.set_xlabel("Dominant Change")
    ax.set_ylabel("Source")
    plt.tight_layout()
    fig.savefig(FIG_DIR / "fig5_heatmap_source_change.png", dpi=150)
    plt.close()
    print("  Fig 5: heatmap\n")

# ── 4. Chi-square tests ────────────────────────────────────────────────────────

def chi_square_tests(flat: pd.DataFrame) -> list:
    print("── 4. Chi-square tests ──")
    valid = ["strengthened", "weakened", "unchanged", "removed", "added"]
    flat_v = flat[flat["change_type"].isin(valid)].copy()
    results = []

    for group_col in ["source", "venue_type"]:
        ct_table = pd.crosstab(flat_v[group_col], flat_v["change_type"])
        chi2, p, dof, _ = stats.chi2_contingency(ct_table)
        results.append({
            "test":  f"change_type ~ {group_col}",
            "chi2":  round(chi2, 2),
            "df":    dof,
            "pvalue": round(p, 6),
        })
        sig = "***" if p < 0.001 else ("**" if p < 0.01 else ("*" if p < 0.05 else "ns"))
        print(f"  {group_col}: χ²={chi2:.1f}, df={dof}, p={p:.2e} {sig}")

    print()
    return results

# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    print("=== Step 10: Statistical Analysis ===\n")

    flat, pairs, corpus = load_data()

    stats_df = descriptive_stats(flat, pairs)
    chi_results = chi_square_tests(flat)
    reg_df   = run_regressions(flat, pairs)
    make_figures(flat, pairs)

    # ── Write report ──────────────────────────────────────────────────────────
    lines = [
        "=== Step 10: Statistical Analysis Report ===\n",
        f"Claim comparisons analysed : {len(flat):,}",
        f"Pairs analysed             : {len(pairs):,}",
        "\n── Chi-square tests ──",
    ]
    for r in chi_results:
        lines.append(
            f"  {r['test']}: χ²={r['chi2']}, df={r['df']}, p={r['pvalue']}"
        )

    lines += ["\n── Logistic regression results ──"]
    for model_name, grp in reg_df.groupby("model"):
        lines.append(f"\n  Model: {model_name}  (n={grp['n'].iloc[0]:,})")
        lines.append(f"  {'Predictor':<30} {'OR':>8} {'p':>8}")
        lines.append("  " + "-"*50)
        for _, row in grp.iterrows():
            sig = ("***" if row["pvalue"] < 0.001
                   else ("**" if row["pvalue"] < 0.01
                         else ("*" if row["pvalue"] < 0.05 else "")))
            lines.append(
                f"  {row['predictor']:<30} {row['OR']:>8.3f} {row['pvalue']:>8.4f} {sig}"
            )

    lines += [
        "\n── Figures saved ──",
        f"  {FIG_DIR}",
        "  fig1_overall_change_distribution.png",
        "  fig2_change_by_source.png",
        "  fig3_temporal_trend.png",
        "  fig4_change_by_venue.png",
        "  fig5_heatmap_source_change.png",
    ]

    report = "\n".join(lines)
    print("\n" + report)
    OUT_REPORT.write_text(report, encoding="utf-8")
    print(f"\n  Saved report → {OUT_REPORT.name}")
    print("\nStep 10 complete.")


if __name__ == "__main__":
    main()

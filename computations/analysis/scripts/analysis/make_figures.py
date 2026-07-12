"""
make_figures.py  —  Individual panel figures (no subplots)

Each function produces one standalone PNG file.
All panels use a shared colour palette and data loading routine.

Panels produced:
  fig_1a_claim_change_donut.png       — Overall claim-level change (donut)
  fig_1b_hedging_shift_donut.png      — Hedging shift (cautious/confident/no-shift)
  fig_1c_hedging_by_stratum.png       — Hedging composition by content-change stratum
  fig_1d_change_over_time.png         — Dominant change over time (pair level)
  fig_1e_hedging_by_field.png         — Hedging asymmetry by field
  fig_1f_claim_type_alluvial.png      — Claim-type alluvial (preprint → published)

  fig_2a_source_heatmap.png           — Source × dominant-change heatmap
  fig_2b_revision_by_claim_type.png   — Revision by claim type (dot plot)
  fig_2c_change_rates_over_time.png   — Claim change rates over time
  fig_2d_weakening_vs_years.png       — Weakening rate vs years-to-publication
  fig_2e_sw_ratio_by_venue.png        — Strengthened:Weakened ratio by venue
  fig_2f_change_by_source_venue.png   — Change by source × venue (stacked bar)

  fig_3a_journal_prestige_scatter.png — Journal prestige vs weakening rate
  fig_3b_citation_quartile.png        — Dominant change by citation quartile
  fig_3c_open_access.png              — Dominant change by open-access status

Outputs → OUT_DIR (default: computations/analysis/outputs/figures/)

New project layout (relative to project root):
  computations/data/data_sources/processed/analysis_corpus.csv
  computations/data/data_sources/claims/claim_changes.jsonl
  computations/data/data_sources/claims/claims_extracted.jsonl
  computations/analysis/scripts/analysis/make_figures.py   ← this file
  computations/analysis/outputs/figures/                   ← output
"""

from pathlib import Path
import json, re, warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyArrowPatch
from scipy import stats
from collections import Counter

warnings.filterwarnings("ignore")

# ── Paths ──────────────────────────────────────────────────────────────────────
# This script lives at:
#   <project_root>/computations/analysis/scripts/analysis/make_figures.py
# So SCRIPT_DIR.parents[3] is the project root.
SCRIPT_DIR   = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[3]   # .../The Evolution of Scientific Claims...

CORPUS_FILE  = PROJECT_ROOT / "computations" / "data" / "data_sources" / "processed" / "analysis_corpus.csv"
COMP_FILE    = PROJECT_ROOT / "computations" / "data" / "data_sources" / "claims"    / "claim_changes.jsonl"
EXTR_FILE    = PROJECT_ROOT / "computations" / "data" / "data_sources" / "claims"    / "claims_extracted.jsonl"
OUT_DIR      = PROJECT_ROOT / "computations" / "analysis" / "outputs" / "figures"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ── Colour palette ─────────────────────────────────────────────────────────────
C = {
    "unchanged":    "#7fb3d3",
    "weakened":     "#e74c3c",
    "strengthened": "#2ecc71",
    "removed":      "#f39c12",
    "added":        "#9b59b6",
    "mixed":        "#95a5a6",
}
CHANGE_ORDER  = ["strengthened", "weakened", "unchanged", "removed", "added"]

# ── Figure style ───────────────────────────────────────────────────────────────
STYLE = {
    "font.family":   "DejaVu Sans",
    "axes.spines.top":   False,
    "axes.spines.right": False,
    "figure.facecolor":  "white",
    "axes.facecolor":    "white",
}

def apply_style():
    plt.rcParams.update(STYLE)

def save(fig, name):
    path = OUT_DIR / name
    fig.savefig(path, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  Saved → {name}")
    return path

def pct(n, total):
    return n / total * 100 if total else 0


# ══════════════════════════════════════════════════════════════════════════════
# DATA LOADING
# ══════════════════════════════════════════════════════════════════════════════

def load_data():
    """Load corpus, claim-level comparisons, and claim-type data."""
    corpus = pd.read_csv(CORPUS_FILE, low_memory=False)
    corpus["preprint_year"] = pd.to_numeric(corpus["preprint_year"], errors="coerce")
    corpus["years_to_pub"]  = pd.to_numeric(corpus["years_to_pub"],  errors="coerce")

    # Fix arxiv_id float → string
    corpus["arxiv_id_str"] = corpus["arxiv_id"].apply(
        lambda x: f"{x:.5f}".rstrip("0").rstrip(".") if pd.notna(x) else None
    )

    # ── Parse claim_changes.jsonl ──────────────────────────────────────────────
    claim_rows = []
    pair_rows  = []
    with open(COMP_FILE) as f:
        for line in f:
            try:
                rec = json.loads(line)
                pid = str(rec.get("pair_id", ""))
                try:
                    py = int(float(str(rec.get("preprint_year") or 0)))
                except (ValueError, TypeError):
                    py = None
                try:
                    puby = int(float(str(rec.get("pub_year") or 0)))
                except (ValueError, TypeError):
                    puby = None
                pair_rows.append({
                    "pair_id":         pid,
                    "source":          rec.get("source", ""),
                    "preprint_year":   py,
                    "pub_year":        puby,
                    "venue_type":      rec.get("venue_type", ""),
                    "dominant_change": rec.get("dominant_change", ""),
                    "n_strengthened":  rec.get("n_strengthened", 0) or 0,
                    "n_weakened":      rec.get("n_weakened", 0) or 0,
                    "n_unchanged":     rec.get("n_unchanged", 0) or 0,
                    "n_added":         rec.get("n_added", 0) or 0,
                    "n_removed":       rec.get("n_removed", 0) or 0,
                })
                for comp in rec.get("comparisons", []):
                    claim_rows.append({
                        "pair_id":         pid,
                        "source":          rec.get("source", ""),
                        "preprint_year":   py,
                        "venue_type":      rec.get("venue_type", ""),
                        "dominant_change": rec.get("dominant_change", ""),
                        "change_type":     comp.get("change_type", ""),
                        "preprint_claim":  (comp.get("preprint_claim") or "")[:80],
                        "pub_claim":       (comp.get("publication_claim") or "")[:80],
                    })
            except Exception:
                continue

    claims   = pd.DataFrame(claim_rows)
    pairs_df = pd.DataFrame(pair_rows)

    for col in ["dominant_change", "change_type"]:
        if col in claims.columns:
            claims[col] = claims[col].str.strip().str.lower()
    pairs_df["dominant_change"] = pairs_df["dominant_change"].str.strip().str.lower()

    # ── Merge dominant_change into corpus ──────────────────────────────────────
    pairs_df["pub_doi_norm"] = pairs_df["pub_doi"].str.strip().str.lower() if "pub_doi" in pairs_df.columns else ""
    corpus["pub_doi_norm"]   = corpus["pub_doi"].str.strip().str.lower()

    arxiv_map = pairs_df[pairs_df["source"] == "arxiv"].set_index("pair_id")[
        ["dominant_change", "n_strengthened", "n_weakened", "n_unchanged",
         "n_added", "n_removed", "venue_type"]
    ]
    corpus = corpus.merge(
        arxiv_map.rename(columns={"venue_type": "venue_type_cc"}),
        left_on="arxiv_id_str", right_index=True, how="left"
    )
    bio_map = (pairs_df[pairs_df["source"].isin(["biorxiv", "medrxiv"])]
               .drop_duplicates("pub_doi_norm")
               .set_index("pub_doi_norm")[
                   ["dominant_change", "n_strengthened", "n_weakened", "n_unchanged",
                    "n_added", "n_removed", "venue_type"]
               ])
    mask = corpus["dominant_change"].isna()
    filled = corpus.loc[mask, "pub_doi_norm"].map(bio_map["dominant_change"])
    corpus.loc[mask, "dominant_change"] = filled
    for col in ["n_strengthened", "n_weakened", "n_unchanged", "n_added", "n_removed"]:
        corpus.loc[mask, col] = corpus.loc[mask, "pub_doi_norm"].map(bio_map[col])

    if "venue_type" not in corpus.columns:
        corpus["venue_type"] = np.nan
    mask2 = corpus["venue_type"].isna()
    corpus.loc[mask2, "venue_type"] = corpus.loc[mask2, "venue_type_cc"]

    # ── Load claim types from claims_extracted.jsonl ───────────────────────────
    pre_type_lookup = {}
    pub_type_lookup = {}
    with open(EXTR_FILE) as f:
        for line in f:
            try:
                rec = json.loads(line)
                pid = str(rec.get("pair_id", ""))
                pre_type_lookup[pid] = {
                    cl["claim"][:80]: cl.get("type", "unknown")
                    for cl in (rec.get("preprint_claims") or [])
                    if isinstance(cl, dict)
                }
                pub_type_lookup[pid] = {
                    cl["claim"][:80]: cl.get("type", "unknown")
                    for cl in (rec.get("publication_claims") or [])
                    if isinstance(cl, dict)
                }
            except Exception:
                continue

    claims["pre_type"] = claims.apply(
        lambda r: pre_type_lookup.get(r["pair_id"], {}).get(r["preprint_claim"], "unknown"), axis=1
    )
    claims["pub_type"] = claims.apply(
        lambda r: pub_type_lookup.get(r["pair_id"], {}).get(r["pub_claim"], "unknown"), axis=1
    )

    # ── Hedging-shift column ───────────────────────────────────────────────────
    def hedging_shift(row):
        nw = row.get("n_weakened", 0) or 0
        ns = row.get("n_strengthened", 0) or 0
        if nw > ns:   return "more_cautious"
        elif ns > nw: return "more_confident"
        else:         return "no_shift"

    pairs_df["hedging_shift"] = pairs_df.apply(hedging_shift, axis=1)

    # ── Primary field from corpus ──────────────────────────────────────────────
    def primary_field(fstr):
        if pd.isna(fstr):
            return None
        for p in str(fstr).split(";"):
            p = p.strip()
            m = re.match(r"^(\d{2})\s+(.+)", p)
            if m:
                return m.group(2).strip()
        return str(fstr).split(";")[0].strip()[:50]

    corpus["primary_field"] = corpus["field_of_research"].apply(primary_field)

    return corpus, claims, pairs_df


# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 1a — Overall claim-level change donut
# ══════════════════════════════════════════════════════════════════════════════

def fig_1a(corpus, claims, pairs_df):
    apply_style()
    fig, ax = plt.subplots(figsize=(6, 6))
    ct = claims["change_type"].value_counts()
    total = ct.sum()
    labels_order = ["unchanged", "weakened", "strengthened", "removed", "added"]
    vals   = [ct.get(l, 0) for l in labels_order]
    colors = [C[l] for l in labels_order]
    wedges, texts = ax.pie(
        vals, colors=colors, startangle=90,
        wedgeprops=dict(width=0.45, edgecolor="white", linewidth=1.5),
        counterclock=False
    )
    ax.text(0, 0, f"{pct(ct.get('unchanged',0), total):.1f}%\nunchanged",
            ha="center", va="center", fontsize=13, fontweight="bold")
    legend_labels = [f"{l.capitalize()} ({pct(ct.get(l,0),total):.1f}%, n={ct.get(l,0):,})"
                     for l in labels_order]
    ax.legend(wedges, legend_labels, loc="lower center", bbox_to_anchor=(0.5, -0.12),
              fontsize=9, frameon=False, ncol=2)
    ax.set_title("Claim-level change from preprint to publication\n"
                 f"(n = {total:,} comparisons)", fontsize=11, fontweight="bold", pad=10)
    return save(fig, "fig_1a_claim_change_donut.png")


# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 1b — Hedging shift donut
# ══════════════════════════════════════════════════════════════════════════════

def fig_1b(corpus, claims, pairs_df):
    apply_style()
    hs = pairs_df["hedging_shift"].value_counts()
    total = hs.sum()
    labels_order = ["more_cautious", "no_shift", "more_confident"]
    hs_colors    = ["#e74c3c", "#bdc3c7", "#2ecc71"]
    display      = ["More cautious", "No shift", "More confident"]
    vals = [hs.get(l, 0) for l in labels_order]

    fig, ax = plt.subplots(figsize=(6, 6))
    wedges, texts = ax.pie(
        vals, colors=hs_colors, startangle=90,
        wedgeprops=dict(width=0.45, edgecolor="white", linewidth=1.5),
        counterclock=False
    )
    n_caut = hs.get("more_cautious", 0)
    n_conf = hs.get("more_confident", 0)
    ratio  = n_caut / n_conf if n_conf else float("inf")
    ax.text(0, 0, f"{ratio:.1f}:1\ncautious/\nconfident",
            ha="center", va="center", fontsize=11, fontweight="bold")
    legend_labels = [f"{d} ({pct(v,total):.1f}%, n={v:,})"
                     for d, v in zip(display, vals)]
    ax.legend(wedges, legend_labels, loc="lower center", bbox_to_anchor=(0.5, -0.10),
              fontsize=9, frameon=False)
    ax.set_title("Hedging shift per paper\n"
                 f"(n = {total:,} pairs)", fontsize=11, fontweight="bold", pad=10)
    return save(fig, "fig_1b_hedging_shift_donut.png")


# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 1c — Hedging composition by content-change stratum
# ══════════════════════════════════════════════════════════════════════════════

def fig_1c(corpus, claims, pairs_df):
    apply_style()
    def stratum(row):
        nw = row.get("n_weakened", 0) or 0
        ns = row.get("n_strengthened", 0) or 0
        nu = row.get("n_unchanged", 0) or 0
        total = nw + ns + nu
        if total == 0:
            return None
        changed_frac = (nw + ns) / total
        if changed_frac < 0.1:
            return "Unchanged\n(<10% changed)"
        elif changed_frac < 0.4:
            return "Minor change\n(10–40%)"
        else:
            return "Major change\n(>40%)"

    pairs_df["stratum"] = pairs_df.apply(stratum, axis=1)
    strat_order = ["Unchanged\n(<10% changed)", "Minor change\n(10–40%)", "Major change\n(>40%)"]
    hs_colors   = {"more_cautious": "#e74c3c", "no_shift": "#bdc3c7", "more_confident": "#2ecc71"}

    fig, ax = plt.subplots(figsize=(7, 5))
    bottom = np.zeros(3)
    for hs_val, color in hs_colors.items():
        vals = []
        for s in strat_order:
            sub = pairs_df[pairs_df["stratum"] == s]
            vals.append(pct((sub["hedging_shift"] == hs_val).sum(), len(sub)))
        ax.bar(range(3), vals, bottom=bottom, color=color,
               label=hs_val.replace("_", " ").capitalize(), width=0.5)
        bottom += np.array(vals)

    ax.set_xticks(range(3))
    ax.set_xticklabels(strat_order, fontsize=9)
    ax.set_ylabel("% of pairs", fontsize=10)
    ax.set_ylim(0, 105)
    ax.set_title("Hedging shift by content-change stratum", fontsize=11, fontweight="bold")
    ax.legend(fontsize=9, frameon=False, loc="upper right")
    return save(fig, "fig_1c_hedging_by_stratum.png")


# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 1d — Dominant change over time (pair level)
# ══════════════════════════════════════════════════════════════════════════════

def fig_1d(corpus, claims, pairs_df):
    apply_style()
    years = sorted(pairs_df["preprint_year"].dropna().astype(int).unique())
    years = [y for y in years if 2015 <= y <= 2024]
    plot_changes = ["unchanged", "weakened", "strengthened", "mixed"]
    line_colors  = {
        "unchanged":    "#7fb3d3",
        "weakened":     "#e74c3c",
        "strengthened": "#2ecc71",
        "mixed":        "#95a5a6",
    }

    fig, ax = plt.subplots(figsize=(8, 5))
    for change in plot_changes:
        vals = []
        for y in years:
            sub = pairs_df[pairs_df["preprint_year"] == y]
            vals.append(pct((sub["dominant_change"] == change).sum(), len(sub)))
        ax.plot(years, vals, marker="o", markersize=5, linewidth=2,
                color=line_colors[change], label=change.capitalize())

    ax.set_xlabel("Preprint year", fontsize=10)
    ax.set_ylabel("% of pairs", fontsize=10)
    ax.set_title("Dominant change type over time (pair level)", fontsize=11, fontweight="bold")
    ax.legend(fontsize=9, frameon=False)
    ax.set_xticks(years)
    ax.tick_params(axis="x", rotation=45)
    return save(fig, "fig_1d_change_over_time.png")


# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 1e — Hedging asymmetry by field
# ══════════════════════════════════════════════════════════════════════════════

def fig_1e(corpus, claims, pairs_df):
    """Hedging asymmetry by preprint source (arXiv / bioRxiv / medRxiv).
    field_of_research is only populated for biorxiv/medrxiv rows in the corpus,
    so we use source as the grouping variable instead.
    """
    apply_style()
    source_order = ["arxiv", "biorxiv", "medrxiv"]
    source_labels = {"arxiv": "arXiv", "biorxiv": "bioRxiv", "medrxiv": "medRxiv"}

    stats = []
    for src in source_order:
        sub = pairs_df[pairs_df["source"] == src]
        n = len(sub)
        if n < 10:
            continue
        n_cautious  = (sub["hedging_shift"] == "more_cautious").sum()
        n_confident = (sub["hedging_shift"] == "more_confident").sum()
        n_no_shift  = (sub["hedging_shift"] == "no_shift").sum()
        ratio = n_cautious / n_confident if n_confident else np.nan
        stats.append({
            "source":      source_labels[src],
            "n":           n,
            "pct_cautious":  pct(n_cautious, n),
            "pct_confident": pct(n_confident, n),
            "pct_no_shift":  pct(n_no_shift, n),
            "ratio":       ratio,
        })

    fig, ax = plt.subplots(figsize=(7, 5))
    x = np.arange(len(stats))
    width = 0.28
    labels_src = [s["source"] for s in stats]

    bars_c = ax.bar(x - width, [s["pct_cautious"]  for s in stats], width,
                    color="#e74c3c", label="More cautious", alpha=0.85)
    bars_n = ax.bar(x,          [s["pct_no_shift"]  for s in stats], width,
                    color="#bdc3c7", label="No shift",      alpha=0.85)
    bars_s = ax.bar(x + width,  [s["pct_confident"] for s in stats], width,
                    color="#2ecc71", label="More confident", alpha=0.85)

    # Annotate cautious:confident ratio
    for i, s in enumerate(stats):
        ax.text(i, s["pct_cautious"] + 0.8, f"{s['ratio']:.2f}:1",
                ha="center", va="bottom", fontsize=9, color="#e74c3c", fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels([f"{s['source']}\n(n={s['n']:,})" for s in stats], fontsize=10)
    ax.set_ylabel("% of papers", fontsize=10)
    ax.set_title("Hedging shift by preprint source\n(cautious:confident ratio annotated above each group)",
                 fontsize=11, fontweight="bold")
    ax.legend(fontsize=9, frameon=False, loc="upper right")
    ax.set_ylim(0, max(s["pct_cautious"] for s in stats) * 1.25)
    return save(fig, "fig_1e_hedging_by_field.png")


# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 1f — Claim-type alluvial (preprint → published)
# ══════════════════════════════════════════════════════════════════════════════

def fig_1f(corpus, claims, pairs_df):
    apply_style()
    from matplotlib.path import Path as MPath
    import matplotlib.patches as mpatches

    MAIN_TYPES = ["result", "contribution", "conclusion", "limitation"]
    pre_counts = Counter(t for t in claims["pre_type"] if t in MAIN_TYPES)
    pub_counts = Counter(t for t in claims["pub_type"] if t in MAIN_TYPES)
    all_types  = [t for t in MAIN_TYPES if pre_counts.get(t, 0) + pub_counts.get(t, 0) > 0]

    type_colors = {
        "result":       "#3498db",
        "contribution": "#2ecc71",
        "conclusion":   "#e74c3c",
        "limitation":   "#f39c12",
    }

    fig, ax = plt.subplots(figsize=(10, 7))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    pre_total = sum(pre_counts.get(t, 0) for t in all_types) or 1
    pub_total = sum(pub_counts.get(t, 0) for t in all_types) or 1
    bar_width = 0.10
    gap = 0.008
    usable_height = 0.82
    y_start = 0.06

    # Compute bar positions
    pre_y, pub_y = {}, {}
    y = y_start
    for t in all_types:
        h = pre_counts.get(t, 0) / pre_total * usable_height
        pre_y[t] = (y, y + h)
        y += h + gap

    y = y_start
    for t in all_types:
        h = pub_counts.get(t, 0) / pub_total * usable_height
        pub_y[t] = (y, y + h)
        y += h + gap

    bar_x_pre = 0.22
    bar_x_pub = 0.68

    # Draw ribbons first (behind bars)
    for t in all_types:
        if t not in pre_y or t not in pub_y:
            continue
        color = type_colors.get(t, "#95a5a6")
        y0_bot, y0_top = pre_y[t]
        y1_bot, y1_top = pub_y[t]
        x0 = bar_x_pre + bar_width
        x1 = bar_x_pub
        cx = (x0 + x1) / 2
        verts = [
            (x0, y0_bot), (cx, y0_bot), (cx, y1_bot), (x1, y1_bot),
            (x1, y1_top), (cx, y1_top), (cx, y0_top), (x0, y0_top),
            (x0, y0_bot),
        ]
        codes = [
            MPath.MOVETO, MPath.CURVE4, MPath.CURVE4, MPath.CURVE4,
            MPath.LINETO, MPath.CURVE4, MPath.CURVE4, MPath.CURVE4,
            MPath.CLOSEPOLY,
        ]
        # Use simpler bezier: MOVETO, LINETO approach with fill
        from matplotlib.patches import PathPatch
        # Draw as filled polygon with bezier-like shape using 4-point bezier
        t_vals = np.linspace(0, 1, 50)
        # Top edge: bezier from (x0,y0_top) to (x1,y1_top)
        top_x = (1-t_vals)**3*x0 + 3*(1-t_vals)**2*t_vals*cx + 3*(1-t_vals)*t_vals**2*cx + t_vals**3*x1
        top_y = (1-t_vals)**3*y0_top + 3*(1-t_vals)**2*t_vals*y0_top + 3*(1-t_vals)*t_vals**2*y1_top + t_vals**3*y1_top
        # Bottom edge: bezier from (x1,y1_bot) back to (x0,y0_bot)
        bot_x = (1-t_vals)**3*x1 + 3*(1-t_vals)**2*t_vals*cx + 3*(1-t_vals)*t_vals**2*cx + t_vals**3*x0
        bot_y = (1-t_vals)**3*y1_bot + 3*(1-t_vals)**2*t_vals*y1_bot + 3*(1-t_vals)*t_vals**2*y0_bot + t_vals**3*y0_bot
        poly_x = np.concatenate([top_x, bot_x])
        poly_y = np.concatenate([top_y, bot_y])
        ax.fill(poly_x, poly_y, color=color, alpha=0.30, zorder=1)

    # Draw bars on top of ribbons
    for t in all_types:
        color = type_colors.get(t, "#95a5a6")
        y0, y1 = pre_y[t]
        h = y1 - y0
        ax.add_patch(plt.Rectangle((bar_x_pre, y0), bar_width, h, color=color, alpha=0.90, zorder=2))
        # Left labels
        ax.text(bar_x_pre - 0.01, y0 + h/2,
                f"{t}\n{pre_counts.get(t,0):,}",
                ha="right", va="center", fontsize=8.5, color=color, fontweight="bold")

    for t in all_types:
        color = type_colors.get(t, "#95a5a6")
        y0, y1 = pub_y[t]
        h = y1 - y0
        ax.add_patch(plt.Rectangle((bar_x_pub, y0), bar_width, h, color=color, alpha=0.90, zorder=2))
        # Right labels
        ax.text(bar_x_pub + bar_width + 0.01, y0 + h/2,
                f"{t}\n{pub_counts.get(t,0):,}",
                ha="left", va="center", fontsize=8.5, color=color, fontweight="bold")

    # Column headers
    ax.text(bar_x_pre + bar_width/2, 0.95, "Preprint",
            ha="center", va="bottom", fontsize=12, fontweight="bold")
    ax.text(bar_x_pub + bar_width/2, 0.95, "Published",
            ha="center", va="bottom", fontsize=12, fontweight="bold")
    ax.set_title("Claim-type distribution: preprint vs. published",
                 fontsize=12, fontweight="bold", y=1.01)
    return save(fig, "fig_1f_claim_type_alluvial.png")


# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 2a — Source × dominant-change heatmap
# ══════════════════════════════════════════════════════════════════════════════

def fig_2a(corpus, claims, pairs_df):
    apply_style()
    import matplotlib.colors as mcolors

    source_order = ["arxiv", "biorxiv", "medrxiv"]
    change_order = ["unchanged", "weakened", "strengthened", "mixed", "removed"]
    mat = np.zeros((len(source_order), len(change_order)))
    for i, src in enumerate(source_order):
        sub = pairs_df[pairs_df["source"] == src]
        for j, ch in enumerate(change_order):
            mat[i, j] = pct((sub["dominant_change"] == ch).sum(), len(sub))

    fig, ax = plt.subplots(figsize=(7, 4))
    im = ax.imshow(mat, cmap="Blues", aspect="auto", vmin=0, vmax=mat.max())
    ax.set_xticks(range(len(change_order)))
    ax.set_xticklabels([c.capitalize() for c in change_order], fontsize=9)
    ax.set_yticks(range(len(source_order)))
    ax.set_yticklabels([s.capitalize() for s in source_order], fontsize=9)
    for i in range(len(source_order)):
        for j in range(len(change_order)):
            ax.text(j, i, f"{mat[i,j]:.1f}%", ha="center", va="center",
                    fontsize=9, color="white" if mat[i,j] > mat.max() * 0.6 else "black")
    plt.colorbar(im, ax=ax, label="% of pairs", shrink=0.8)
    ax.set_title("Dominant change type by preprint source", fontsize=11, fontweight="bold")
    return save(fig, "fig_2a_source_heatmap.png")


# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 2b — Revision by claim type (dot plot)
# ══════════════════════════════════════════════════════════════════════════════

def fig_2b(corpus, claims, pairs_df):
    apply_style()
    MAIN_TYPES = ["result", "contribution", "conclusion", "limitation", "method"]
    type_stats = []
    for t in MAIN_TYPES:
        sub = claims[claims["pre_type"] == t]
        if len(sub) < 100:
            continue
        n = len(sub)
        type_stats.append({
            "type":         t,
            "n":            n,
            "pct_weakened": pct((sub["change_type"] == "weakened").sum(), n),
            "pct_strengthened": pct((sub["change_type"] == "strengthened").sum(), n),
        })
    type_stats = sorted(type_stats, key=lambda x: x["pct_weakened"])

    fig, ax = plt.subplots(figsize=(7, max(4, len(type_stats) * 0.7)))
    for i, row in enumerate(type_stats):
        xw = row["pct_weakened"]
        xs = row["pct_strengthened"]
        ax.plot(xw, i, "o", color="#e74c3c", markersize=9, zorder=3)
        ax.plot(xs, i, "o", color="#2ecc71", markersize=9, zorder=3)
        ax.plot([xs, xw], [i, i], color="#cccccc", linewidth=1.5, zorder=2)
        ax.text(xw + 0.3, i, f"{xw:.1f}%", va="center", fontsize=8, color="#e74c3c")
        ax.text(xs - 0.3, i, f"{xs:.1f}%", va="center", fontsize=8, color="#2ecc71", ha="right")

    ax.set_yticks(range(len(type_stats)))
    ax.set_yticklabels([f"{r['type'].capitalize()} (n={r['n']:,})" for r in type_stats], fontsize=9)
    ax.set_xlabel("% of claims", fontsize=10)
    ax.set_title("Weakened vs. strengthened rate by claim type", fontsize=11, fontweight="bold")
    legend_handles = [
        mpatches.Patch(color="#e74c3c", label="Weakened"),
        mpatches.Patch(color="#2ecc71", label="Strengthened"),
    ]
    ax.legend(handles=legend_handles, fontsize=9, frameon=False)
    return save(fig, "fig_2b_revision_by_claim_type.png")


# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 2c — Claim change rates over time
# ══════════════════════════════════════════════════════════════════════════════

def fig_2c(corpus, claims, pairs_df):
    apply_style()
    years = sorted(claims["preprint_year"].dropna().astype(int).unique())
    years = [y for y in years if 2015 <= y <= 2024]
    plot_changes = ["unchanged", "weakened", "strengthened"]
    line_colors  = {"unchanged": "#7fb3d3", "weakened": "#e74c3c", "strengthened": "#2ecc71"}

    fig, ax = plt.subplots(figsize=(8, 5))
    for change in plot_changes:
        vals = []
        for y in years:
            sub = claims[claims["preprint_year"] == y]
            vals.append(pct((sub["change_type"] == change).sum(), len(sub)))
        ax.plot(years, vals, marker="o", markersize=5, linewidth=2,
                color=line_colors[change], label=change.capitalize())

    ax.set_xlabel("Preprint year", fontsize=10)
    ax.set_ylabel("% of claim comparisons", fontsize=10)
    ax.set_title("Claim change rates over time", fontsize=11, fontweight="bold")
    ax.legend(fontsize=9, frameon=False)
    ax.set_xticks(years)
    ax.tick_params(axis="x", rotation=45)
    return save(fig, "fig_2c_change_rates_over_time.png")


# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 2d — Weakening rate vs years-to-publication
# ══════════════════════════════════════════════════════════════════════════════

def fig_2d(corpus, claims, pairs_df):
    apply_style()
    pairs_df["years_to_pub"] = pairs_df.apply(
        lambda r: (r["pub_year"] - r["preprint_year"])
        if pd.notna(r.get("pub_year")) and pd.notna(r.get("preprint_year")) else np.nan,
        axis=1
    )
    valid = pairs_df[pairs_df["years_to_pub"].between(0, 8)].copy()
    bins  = [0, 0.5, 1, 2, 4, 8]
    labels = ["0–0.5", "0.5–1", "1–2", "2–4", "4–8"]
    valid["ytp_bin"] = pd.cut(valid["years_to_pub"], bins=bins, labels=labels, right=False)

    bin_stats = valid.groupby("ytp_bin", observed=True).agg(
        n=("dominant_change", "count"),
        n_weakened=("dominant_change", lambda x: (x == "weakened").sum()),
        n_strengthened=("dominant_change", lambda x: (x == "strengthened").sum()),
    ).reset_index()
    bin_stats["pct_weakened"]     = bin_stats["n_weakened"] / bin_stats["n"] * 100
    bin_stats["pct_strengthened"] = bin_stats["n_strengthened"] / bin_stats["n"] * 100

    fig, ax = plt.subplots(figsize=(7, 5))
    x = np.arange(len(bin_stats))
    ax.plot(x, bin_stats["pct_weakened"],     "o-", color="#e74c3c", linewidth=2,
            markersize=7, label="Weakened")
    ax.plot(x, bin_stats["pct_strengthened"], "o-", color="#2ecc71", linewidth=2,
            markersize=7, label="Strengthened")

    # Regression on weakened
    if len(x) > 2:
        slope, intercept, r, p, _ = stats.linregress(x, bin_stats["pct_weakened"])
        xfit = np.linspace(x.min(), x.max(), 100)
        ax.plot(xfit, slope * xfit + intercept, "--", color="#c0392b", linewidth=1.2,
                label=f"Trend ($R^2={r**2:.2f}$, $p={p:.3f}$)")

    ax.set_xticks(x)
    ax.set_xticklabels([f"{l}\n(n={n:,})" for l, n in
                        zip(bin_stats["ytp_bin"], bin_stats["n"])], fontsize=8)
    ax.set_xlabel("Years from preprint to publication", fontsize=10)
    ax.set_ylabel("% of pairs", fontsize=10)
    ax.set_title("Weakening rate vs. time to publication", fontsize=11, fontweight="bold")
    ax.legend(fontsize=9, frameon=False)
    return save(fig, "fig_2d_weakening_vs_years.png")


# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 2e — Strengthened:Weakened ratio by venue type
# ══════════════════════════════════════════════════════════════════════════════

def fig_2e(corpus, claims, pairs_df):
    apply_style()
    venue_map = {"journal": "Journal", "conference": "Conference", "book": "Book"}
    venue_labels, sw_ratios, ns = [], [], []
    for vk, vl in venue_map.items():
        sub = pairs_df[pairs_df["venue_type"].str.lower().str.contains(vk, na=False)]
        nw  = (sub["dominant_change"] == "weakened").sum()
        ns_ = (sub["dominant_change"] == "strengthened").sum()
        if nw > 0:
            venue_labels.append(f"{vl}\n(n={len(sub):,})")
            sw_ratios.append(ns_ / nw)
            ns.append(len(sub))

    fig, ax = plt.subplots(figsize=(6, 5))
    colors = ["#e74c3c" if r < 1 else "#2ecc71" for r in sw_ratios]
    bars = ax.bar(range(len(venue_labels)), sw_ratios, color=colors, width=0.5)
    ax.axhline(1.0, color="black", linestyle="--", linewidth=1, label="Parity (1:1)")
    ax.set_xticks(range(len(venue_labels)))
    ax.set_xticklabels(venue_labels, fontsize=9)
    ax.set_ylabel("Strengthened / Weakened ratio", fontsize=10)
    ax.set_title("Strengthened:Weakened ratio by venue type", fontsize=11, fontweight="bold")
    ax.legend(fontsize=9, frameon=False)
    for bar, val in zip(bars, sw_ratios):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                f"{val:.2f}", ha="center", va="bottom", fontsize=9, fontweight="bold")
    return save(fig, "fig_2e_sw_ratio_by_venue.png")


# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 2f — Change by source × venue (stacked bar)
# ══════════════════════════════════════════════════════════════════════════════

def fig_2f(corpus, claims, pairs_df):
    apply_style()
    combos = [
        ("arxiv",    "journal",    "arXiv\nJournal"),
        ("arxiv",    "conference", "arXiv\nConference"),
        ("biorxiv",  "journal",    "bioRxiv\nJournal"),
        ("medrxiv",  "journal",    "medRxiv\nJournal"),
    ]
    x_labels, data = [], {ch: [] for ch in CHANGE_ORDER}
    for src, ven, label in combos:
        sub = pairs_df[
            pairs_df["source"].str.lower().str.contains(src, na=False) &
            pairs_df["venue_type"].str.lower().str.contains(ven, na=False)
        ]
        if len(sub) < 50:
            continue
        x_labels.append(f"{label}\n(n={len(sub):,})")
        for ch in CHANGE_ORDER:
            data[ch].append(pct((sub["dominant_change"] == ch).sum(), len(sub)))

    fig, ax = plt.subplots(figsize=(8, 5))
    bottom = np.zeros(len(x_labels))
    for change in CHANGE_ORDER:
        vals = np.array(data[change])
        ax.bar(range(len(x_labels)), vals, bottom=bottom, color=C[change],
               label=change.capitalize(), width=0.55)
        bottom += vals

    ax.set_xticks(range(len(x_labels)))
    ax.set_xticklabels(x_labels, fontsize=8)
    ax.set_ylabel("% of pairs", fontsize=10)
    ax.set_ylim(0, 105)
    ax.set_title("Dominant change by source × venue", fontsize=11, fontweight="bold")
    handles = [mpatches.Patch(color=C[c], label=c.capitalize()) for c in CHANGE_ORDER]
    ax.legend(handles=handles, fontsize=9, frameon=False, loc="upper right")
    return save(fig, "fig_2f_change_by_source_venue.png")


# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 3a — Journal prestige vs weakening rate
# ══════════════════════════════════════════════════════════════════════════════

def fig_3a(corpus, claims, pairs_df):
    apply_style()
    merged = corpus.merge(
        pairs_df[["pair_id", "dominant_change"]],
        left_on="arxiv_id_str", right_on="pair_id", how="left",
        suffixes=("", "_pairs")
    )
    mask = merged["dominant_change"].isna()
    if "pub_doi" in pairs_df.columns:
        bio_map = (pairs_df[pairs_df["source"].isin(["biorxiv", "medrxiv"])]
                   .drop_duplicates("pub_doi").set_index("pub_doi")["dominant_change"])
        merged.loc[mask, "dominant_change"] = merged.loc[mask, "pub_doi"].map(bio_map)

    journal_stats = merged.groupby("pub_journal").agg(
        n_pairs=("pub_doi", "count"),
        median_citations=("pub_citations_count", "median"),
        weakened_frac=("dominant_change", lambda x: (x == "weakened").sum() / max(len(x), 1)),
    ).reset_index()
    jstat = journal_stats[(journal_stats["n_pairs"] >= 20) &
                          (journal_stats["median_citations"] > 0)].copy()

    x     = np.log10(jstat["median_citations"] + 1)
    y     = jstat["weakened_frac"] * 100
    sizes = np.clip(jstat["n_pairs"] / 5, 10, 200)

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.scatter(x, y, s=sizes, alpha=0.55, color="#e74c3c", edgecolors="none")

    if len(x) > 5:
        slope, intercept, r, p, _ = stats.linregress(x, y)
        xfit = np.linspace(x.min(), x.max(), 100)
        ax.plot(xfit, slope * xfit + intercept, "--", color="#c0392b", linewidth=1.5,
                label=f"$R^2={r**2:.2f}$, $p={p:.3f}$")
        ax.legend(fontsize=9, frameon=False)

    top_j = jstat.nlargest(5, "n_pairs")
    for _, row in top_j.iterrows():
        xi = np.log10(row["median_citations"] + 1)
        yi = row["weakened_frac"] * 100
        ax.annotate(row["pub_journal"][:25], (xi, yi), fontsize=6, alpha=0.7,
                    xytext=(3, 3), textcoords="offset points")

    ax.set_xlabel("Median citations (log₁₀ scale)", fontsize=10)
    ax.set_ylabel("% weakened pairs", fontsize=10)
    ax.set_title("Journal prestige vs. weakening rate\n(bubble size ∝ n pairs)",
                 fontsize=11, fontweight="bold")
    return save(fig, "fig_3a_journal_prestige_scatter.png")


# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 3b — Dominant change by citation quartile
# ══════════════════════════════════════════════════════════════════════════════

def fig_3b(corpus, claims, pairs_df):
    apply_style()
    merged = corpus.merge(
        pairs_df[["pair_id", "dominant_change"]],
        left_on="arxiv_id_str", right_on="pair_id", how="left",
        suffixes=("", "_pairs")
    )
    merged_valid = merged[merged["dominant_change"].notna() &
                          merged["pub_citations_count"].notna()].copy()
    merged_valid["cit_quartile"] = pd.qcut(
        merged_valid["pub_citations_count"], q=4,
        labels=["Q1\n(fewest)", "Q2", "Q3", "Q4\n(most)"]
    )
    q_counts = merged_valid.groupby(["cit_quartile", "dominant_change"], observed=True).size().unstack(fill_value=0)
    q_pct    = q_counts.div(q_counts.sum(axis=1), axis=0) * 100

    fig, ax = plt.subplots(figsize=(7, 5))
    bottom = np.zeros(len(q_pct))
    for change in CHANGE_ORDER:
        if change in q_pct.columns:
            vals = q_pct[change].values
            ax.bar(range(len(q_pct)), vals, bottom=bottom,
                   color=C[change], label=change.capitalize(), width=0.55)
            bottom += vals

    ax.set_xticks(range(len(q_pct)))
    ax.set_xticklabels(q_pct.index.tolist(), fontsize=9)
    ax.set_ylabel("% of pairs", fontsize=10)
    ax.set_ylim(0, 105)
    ax.set_title("Dominant change by citation quartile", fontsize=11, fontweight="bold")
    handles = [mpatches.Patch(color=C[c], label=c.capitalize()) for c in CHANGE_ORDER]
    ax.legend(handles=handles, fontsize=9, frameon=False, loc="upper right")
    return save(fig, "fig_3b_citation_quartile.png")


# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 3c — Dominant change by open-access status
# ══════════════════════════════════════════════════════════════════════════════

def fig_3c(corpus, claims, pairs_df):
    apply_style()
    merged = corpus.merge(
        pairs_df[["pair_id", "dominant_change"]],
        left_on="arxiv_id_str", right_on="pair_id", how="left",
        suffixes=("", "_pairs")
    )
    oa_col = "is_open_access" if "is_open_access" in merged.columns else "pub_open_access"
    if oa_col not in merged.columns:
        print("  Skipping 3c — OA column not found")
        return None

    oa_valid = merged[merged["dominant_change"].notna() & merged[oa_col].notna()].copy()
    oa_valid[oa_col] = oa_valid[oa_col].astype(str).str.strip().str.lower()
    oa_valid = oa_valid[oa_valid[oa_col].isin(["true", "false", "1", "0"])]
    oa_valid["oa_label"] = oa_valid[oa_col].map(
        {"true": "Open Access", "false": "Subscription", "1": "Open Access", "0": "Subscription"}
    )
    oa_counts = oa_valid.groupby(["oa_label", "dominant_change"], observed=True).size().unstack(fill_value=0)
    oa_pct    = oa_counts.div(oa_counts.sum(axis=1), axis=0) * 100

    n_oa  = oa_valid[oa_valid["oa_label"] == "Open Access"].shape[0]
    n_sub = oa_valid[oa_valid["oa_label"] == "Subscription"].shape[0]

    fig, ax = plt.subplots(figsize=(6, 5))
    bottom = np.zeros(len(oa_pct))
    for change in CHANGE_ORDER:
        if change in oa_pct.columns:
            vals = oa_pct[change].values
            ax.bar(range(len(oa_pct)), vals, bottom=bottom,
                   color=C[change], label=change.capitalize(), width=0.45)
            bottom += vals

    # Annotate weakened % inside bar
    for i, label in enumerate(oa_pct.index):
        w_pct = oa_pct.loc[label, "weakened"] if "weakened" in oa_pct.columns else 0
        s_pct = oa_pct.loc[label, "strengthened"] if "strengthened" in oa_pct.columns else 0
        ax.text(i, s_pct + w_pct / 2, f"{w_pct:.1f}%",
                ha="center", va="center", fontsize=9, color="white", fontweight="bold")

    ax.set_xticks(range(len(oa_pct)))
    ax.set_xticklabels([
        f"Open Access\n(n={n_oa:,})",
        f"Subscription\n(n={n_sub:,})"
    ], fontsize=9)
    ax.set_ylabel("% of pairs", fontsize=10)
    ax.set_ylim(0, 105)
    ax.set_title("Dominant change by open-access status", fontsize=11, fontweight="bold")
    handles = [mpatches.Patch(color=C[c], label=c.capitalize()) for c in CHANGE_ORDER]
    ax.legend(handles=handles, fontsize=9, frameon=False, loc="lower right")
    return save(fig, "fig_3c_open_access.png")


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main():
    print("=== Generating individual panel figures ===\n")
    corpus, claims, pairs_df = load_data()
    print(f"  Corpus:   {len(corpus):,} pairs")
    print(f"  Claims:   {len(claims):,} comparisons")
    print(f"  Pairs_df: {len(pairs_df):,} pairs from JSONL\n")

    # Figure 1 panels
    fig_1a(corpus, claims, pairs_df)
    fig_1b(corpus, claims, pairs_df)
    fig_1c(corpus, claims, pairs_df)
    fig_1d(corpus, claims, pairs_df)
    fig_1e(corpus, claims, pairs_df)
    fig_1f(corpus, claims, pairs_df)

    # Figure 2 panels
    fig_2a(corpus, claims, pairs_df)
    fig_2b(corpus, claims, pairs_df)
    fig_2c(corpus, claims, pairs_df)
    fig_2d(corpus, claims, pairs_df)
    fig_2e(corpus, claims, pairs_df)
    fig_2f(corpus, claims, pairs_df)

    # Figure 3 panels
    fig_3a(corpus, claims, pairs_df)
    fig_3b(corpus, claims, pairs_df)
    fig_3c(corpus, claims, pairs_df)

    print(f"\nDone. {len(list(OUT_DIR.glob('*.png')))} panels saved to {OUT_DIR}")


if __name__ == "__main__":
    main()

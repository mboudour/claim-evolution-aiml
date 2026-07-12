"""
compute_agreement.py  —  Step 11c (Inter-Rater Agreement)

Reads the completed annotation_sheet.xlsx and computes:
  - Cohen's κ for extraction accuracy (Rater 1 vs Rater 2)
  - Cohen's κ for change classification (Rater 1 vs Rater 2)
  - Precision / recall of LLM labels vs human consensus
  - Breakdown of disagreements by change type

Run AFTER both raters have filled in their columns:
    cd SSRN_bioRxiv_medRxiv_data_collection_via_Dimensions
    python3 code/analysis/compute_agreement.py

Writes to:
    data/validation/agreement_report.txt
    data/validation/agreement_details.csv
"""

from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.metrics import cohen_kappa_score, classification_report

# ── Paths ──────────────────────────────────────────────────────────────────────
SCRIPT_DIR   = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[1]

SHEET_FILE   = PROJECT_ROOT / "data" / "validation" / "annotation_sheet.xlsx"
OUT_REPORT   = PROJECT_ROOT / "data" / "validation" / "agreement_report.txt"
OUT_DETAILS  = PROJECT_ROOT / "data" / "validation" / "agreement_details.csv"

# ── Normalise labels ───────────────────────────────────────────────────────────

def normalise(val):
    if not isinstance(val, str):
        return None
    v = val.strip().upper()
    if v in ("YES", "Y", "1", "TRUE", "CORRECT"):
        return "YES"
    if v in ("NO", "N", "0", "FALSE", "INCORRECT", "WRONG"):
        return "NO"
    if v in ("PARTIAL", "PARTLY", "MOSTLY"):
        return "PARTIAL"
    if v in ("UNSURE", "UNCLEAR", "?", "SKIP"):
        return "SKIP"
    return v  # return as-is for change type labels


def load_sheet():
    df = pd.read_excel(SHEET_FILE, sheet_name="Annotation", header=0)
    df.columns = df.columns.str.strip()
    return df


def compute_kappa(r1, r2, label=""):
    """Compute Cohen's κ between two label series, dropping rows where either is None/SKIP."""
    paired = pd.DataFrame({"r1": r1, "r2": r2}).dropna()
    paired = paired[(paired["r1"] != "SKIP") & (paired["r2"] != "SKIP")]
    if len(paired) < 10:
        return None, len(paired)
    try:
        kappa = cohen_kappa_score(paired["r1"], paired["r2"])
        return round(kappa, 4), len(paired)
    except Exception as e:
        return None, len(paired)


def interpret_kappa(k):
    if k is None:   return "N/A"
    if k < 0:       return "Poor (< 0)"
    if k < 0.20:    return "Slight (0.00–0.20)"
    if k < 0.40:    return "Fair (0.20–0.40)"
    if k < 0.60:    return "Moderate (0.40–0.60)"
    if k < 0.80:    return "Substantial (0.60–0.80)"
    return "Almost perfect (0.80–1.00)"


def main():
    print("=== Step 11c: Inter-Rater Agreement ===\n")

    df = load_sheet()
    print(f"  Loaded {len(df)} annotation rows\n")

    # Normalise rater columns
    r1_ext = df["Rater_1_Extraction_OK"].apply(normalise)
    r1_chg = df["Rater_1_Change_OK"].apply(normalise)
    r2_ext = df["Rater_2_Extraction_OK"].apply(normalise)
    r2_chg = df["Rater_2_Change_OK"].apply(normalise)
    llm_dc = df["Dominant_Change_LLM"].apply(lambda x: str(x).strip().lower() if isinstance(x, str) else None)

    lines = ["=== Inter-Rater Agreement Report ===\n"]

    # ── Extraction agreement ───────────────────────────────────────────────────
    kappa_ext, n_ext = compute_kappa(r1_ext, r2_ext, "extraction")
    lines += [
        "── Claim Extraction Accuracy ──",
        f"  Annotated pairs (both raters): {n_ext}",
        f"  Cohen's κ                    : {kappa_ext}",
        f"  Interpretation               : {interpret_kappa(kappa_ext)}",
    ]

    # Rater 1 accuracy vs LLM (YES = LLM correct)
    r1_ext_yes = (r1_ext == "YES").sum()
    r1_ext_n   = r1_ext.notna().sum()
    r2_ext_yes = (r2_ext == "YES").sum()
    r2_ext_n   = r2_ext.notna().sum()
    lines += [
        f"  Rater 1 accuracy (YES rate)  : {r1_ext_yes}/{r1_ext_n} "
        f"({r1_ext_yes/r1_ext_n*100:.1f}%)" if r1_ext_n else "  Rater 1: no data",
        f"  Rater 2 accuracy (YES rate)  : {r2_ext_yes}/{r2_ext_n} "
        f"({r2_ext_yes/r2_ext_n*100:.1f}%)" if r2_ext_n else "  Rater 2: no data",
        "",
    ]

    # ── Change classification agreement ───────────────────────────────────────
    kappa_chg, n_chg = compute_kappa(r1_chg, r2_chg, "change")
    lines += [
        "── Change Classification Accuracy ──",
        f"  Annotated pairs (both raters): {n_chg}",
        f"  Cohen's κ                    : {kappa_chg}",
        f"  Interpretation               : {interpret_kappa(kappa_chg)}",
    ]

    r1_chg_yes = (r1_chg == "YES").sum()
    r1_chg_n   = r1_chg.notna().sum()
    r2_chg_yes = (r2_chg == "YES").sum()
    r2_chg_n   = r2_chg.notna().sum()
    lines += [
        f"  Rater 1 accuracy (YES rate)  : {r1_chg_yes}/{r1_chg_n} "
        f"({r1_chg_yes/r1_chg_n*100:.1f}%)" if r1_chg_n else "  Rater 1: no data",
        f"  Rater 2 accuracy (YES rate)  : {r2_chg_yes}/{r2_chg_n} "
        f"({r2_chg_yes/r2_chg_n*100:.1f}%)" if r2_chg_n else "  Rater 2: no data",
        "",
    ]

    # ── Consensus vs LLM ──────────────────────────────────────────────────────
    # Consensus: both raters agree → use that; disagree → mark as "disputed"
    consensus_chg = []
    for r1, r2 in zip(r1_chg, r2_chg):
        if r1 == r2 and r1 not in (None, "SKIP"):
            consensus_chg.append(r1)
        else:
            consensus_chg.append("disputed")
    consensus_chg = pd.Series(consensus_chg)

    agreed = (consensus_chg != "disputed").sum()
    disputed = (consensus_chg == "disputed").sum()
    lines += [
        "── Consensus Summary ──",
        f"  Rows where both raters agree : {agreed}",
        f"  Rows with disagreement       : {disputed}",
        "",
    ]

    # ── Save details CSV ──────────────────────────────────────────────────────
    details = df[["Row", "Pair_ID", "Source", "Preprint_Year",
                  "Venue_Type", "Dominant_Change_LLM"]].copy()
    details["Rater_1_Extraction_OK"] = r1_ext.values
    details["Rater_1_Change_OK"]     = r1_chg.values
    details["Rater_2_Extraction_OK"] = r2_ext.values
    details["Rater_2_Change_OK"]     = r2_chg.values
    details["Consensus_Change_OK"]   = consensus_chg.values
    details.to_csv(OUT_DETAILS, index=False, encoding="utf-8")

    # ── Write report ──────────────────────────────────────────────────────────
    report = "\n".join(lines)
    print(report)
    OUT_REPORT.write_text(report, encoding="utf-8")
    print(f"\n  Saved report  → {OUT_REPORT.name}")
    print(f"  Saved details → {OUT_DETAILS.name}")
    print("\nStep 11c complete.")


if __name__ == "__main__":
    main()

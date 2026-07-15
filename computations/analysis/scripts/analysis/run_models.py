#!/usr/bin/env python3
"""
run_models.py
Step 5/6 of the Claim Evolution Pipeline.

Fits a suite of regression models to test the five core hypotheses:

  H1 (Venue Prestige):    Higher-prestige venues → more claim tempering
  H2 (Subfield):          Subfield moderates the direction of change
  H3 (Temporal):          Tempering rate increases over time (2014–2024)
  H4 (Version Proxy):     VoR-linked pairs show more tempering than green OA
  H5 (Claim Type):        Contribution claims change more than background claims

Models:
  - Logistic regression: P(any_change) ~ covariates
  - Logistic regression: P(tempered | changed) ~ covariates
  - Logistic regression: P(amplified | changed) ~ covariates
  - Multinomial logistic: semantic_label ~ covariates
  - OLS: pct_changed ~ covariates  (pair-level)

Input:  pair_level_dataset.csv, analysis_dataset.csv
Output: model_results.csv, model_summary.txt
"""

import os
import warnings
import numpy as np
import pandas as pd
from pathlib import Path

PAIR_CSV    = os.environ.get("PAIR_OUT_PATH",   "/home/ubuntu/upload/pair_level_dataset.csv")
ALIGN_CSV   = os.environ.get("ALIGN_OUT_PATH",  "/home/ubuntu/upload/analysis_dataset.csv")
RESULTS_DIR = os.environ.get("RESULTS_DIR",     "/home/ubuntu/upload/results")

os.makedirs(RESULTS_DIR, exist_ok=True)

warnings.filterwarnings('ignore')


def load_data():
    pair_df  = pd.read_csv(PAIR_CSV,  low_memory=False)
    align_df = pd.read_csv(ALIGN_CSV, low_memory=False)
    print(f"Pair-level dataset:      {len(pair_df):,} rows")
    print(f"Alignment-level dataset: {len(align_df):,} rows")
    return pair_df, align_df


def prepare_pair_features(df: pd.DataFrame) -> pd.DataFrame:
    """Encode categorical predictors for pair-level models."""
    df = df.copy()

    # Year (continuous, centred at 2019)
    if 'pub_year_int' in df.columns:
        df['year_c'] = df['pub_year_int'].fillna(df['pub_year_int'].median()) - 2019
    elif 'pub_year' in df.columns:
        df['year_c'] = pd.to_numeric(df['pub_year'], errors='coerce').fillna(2019) - 2019

    # Prestige tier (ordinal 1–4, centred)
    if 'prestige_tier' in df.columns:
        df['prestige_c'] = pd.to_numeric(df['prestige_tier'], errors='coerce').fillna(2) - 2.5

    # Version proxy (dummy, reference = VoR_published)
    if 'version_proxy' in df.columns:
        df['is_green_oa']  = (df['version_proxy'] == 'green_OA').astype(int)
        df['is_vor_closed'] = (df['version_proxy'] == 'VoR_closed').astype(int)

    # Venue type (dummy, reference = journal)
    if 'venue_type' in df.columns:
        df['is_conference'] = (df['venue_type'].str.lower().str.contains('conf', na=False)).astype(int)

    # Subfield (dummy, reference = Machine Learning)
    if 'subfield' in df.columns:
        for sf in ['Computer Vision', 'NLP', 'AI & Knowledge', 'Robotics',
                   'Signal Processing', 'Biomedical AI', 'HCI', 'Speech & Audio',
                   'Systems', 'Theory', 'Multimodal', 'Other']:
            col = 'sf_' + sf.lower().replace(' ', '_').replace('&', 'and')
            df[col] = (df['subfield'] == sf).astype(int)

    # Log citation count (pub)
    if 'pub_citation_count' in df.columns:
        df['log_pub_cites'] = np.log1p(pd.to_numeric(df['pub_citation_count'], errors='coerce').fillna(0))

    # Number of claims (log)
    if 'n_alignments' in df.columns:
        df['log_n_claims'] = np.log1p(df['n_alignments'])

    return df


def run_logistic_models(df: pd.DataFrame, results: list):
    """Fit binary logistic models at the pair level."""
    try:
        from sklearn.linear_model import LogisticRegression
        from sklearn.preprocessing import StandardScaler
        from sklearn.metrics import roc_auc_score
        from sklearn.model_selection import cross_val_score
    except ImportError:
        print("scikit-learn not available; skipping logistic models")
        return

    feature_cols = [c for c in [
        'year_c', 'prestige_c', 'is_green_oa', 'is_vor_closed',
        'is_conference', 'log_pub_cites', 'log_n_claims',
        'sf_computer_vision', 'sf_nlp', 'sf_ai_and_knowledge',
        'sf_robotics', 'sf_signal_processing', 'sf_biomedical_ai',
    ] if c in df.columns]

    outcomes = {
        'any_change':    ('pct_changed',  lambda x: (x > 0).astype(int)),
        'tempered':      ('pct_tempered', lambda x: (x > 0).astype(int)),
        'amplified':     ('pct_amplified', lambda x: (x > 0).astype(int)),
    }

    for outcome_name, (col, transform) in outcomes.items():
        if col not in df.columns:
            continue
        y = transform(df[col].fillna(0))
        X = df[feature_cols].fillna(0)

        if y.sum() < 10:
            print(f"  Skipping {outcome_name}: too few positive cases")
            continue

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        model = LogisticRegression(max_iter=1000, random_state=42)
        cv_auc = cross_val_score(model, X_scaled, y, cv=5, scoring='roc_auc').mean()
        model.fit(X_scaled, y)

        for feat, coef in zip(feature_cols, model.coef_[0]):
            results.append({
                'model':     f'logistic_{outcome_name}',
                'outcome':   outcome_name,
                'predictor': feat,
                'coef':      round(coef, 4),
                'cv_auc':    round(cv_auc, 4),
            })
        print(f"  {outcome_name}: CV AUC = {cv_auc:.3f}")


def run_ols_model(df: pd.DataFrame, results: list):
    """Fit OLS model for pct_changed at pair level."""
    try:
        import statsmodels.api as sm
    except ImportError:
        print("statsmodels not available; skipping OLS")
        return

    feature_cols = [c for c in [
        'year_c', 'prestige_c', 'is_green_oa', 'is_vor_closed',
        'is_conference', 'log_pub_cites', 'log_n_claims',
        'sf_computer_vision', 'sf_nlp', 'sf_ai_and_knowledge',
        'sf_robotics',
    ] if c in df.columns]

    if 'pct_changed' not in df.columns:
        return

    y = df['pct_changed'].fillna(0)
    X = sm.add_constant(df[feature_cols].fillna(0))

    model = sm.OLS(y, X).fit(cov_type='HC3')
    summary_path = os.path.join(RESULTS_DIR, 'ols_pct_changed_summary.txt')
    with open(summary_path, 'w') as f:
        f.write(str(model.summary()))
    print(f"  OLS R² = {model.rsquared:.3f}, saved to {summary_path}")

    for feat in feature_cols:
        results.append({
            'model':     'ols_pct_changed',
            'outcome':   'pct_changed',
            'predictor': feat,
            'coef':      round(model.params.get(feat, np.nan), 4),
            'pvalue':    round(model.pvalues.get(feat, np.nan), 4),
            'ci_lower':  round(model.conf_int().loc[feat, 0] if feat in model.conf_int().index else np.nan, 4),
            'ci_upper':  round(model.conf_int().loc[feat, 1] if feat in model.conf_int().index else np.nan, 4),
        })


def main():
    print("Loading data...")
    pair_df, align_df = load_data()

    print("Preparing features...")
    pair_df = prepare_pair_features(pair_df)

    results = []

    print("\nFitting logistic models (pair-level)...")
    run_logistic_models(pair_df, results)

    print("\nFitting OLS model (pair-level)...")
    run_ols_model(pair_df, results)

    # Save results
    if results:
        results_df = pd.DataFrame(results)
        out_path = os.path.join(RESULTS_DIR, 'model_results.csv')
        results_df.to_csv(out_path, index=False)
        print(f"\nSaved model results: {out_path} ({len(results_df):,} rows)")
    else:
        print("\nNo model results to save.")

    print("\n=== MODELLING COMPLETE ===")


if __name__ == '__main__':
    main()

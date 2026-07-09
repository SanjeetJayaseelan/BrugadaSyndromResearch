"""
compute_effect_sizes.py

Regenerates the univariate feature screen (feature_effect_sizes.csv) from the
feature table (features.csv). For each of the 111 features it reports, comparing
BrS vs. control:

  cohens_d  signed Cohen's d, (mean_BrS - mean_control) / pooled SD
  abs_d     |cohens_d|, for ranking
  mwu_p     two-sided Mann-Whitney U p-value
  bonf_sig  True if mwu_p passes a Bonferroni cutoff of 0.05 / n_features

IMPORTANT: this is a *descriptive* screen used only for interpretation and for
annotating figures (fig3, fig6). It is NOT feature selection — train_model.py
fits on all 111 features, so nothing here leaks into the cross-validated
performance numbers. It is provided so the artifact this repo ships
(feature_effect_sizes.csv) is reproducible rather than pre-baked.

Usage:
    python compute_effect_sizes.py --features ../data/features.csv \
        --output ../data/feature_effect_sizes.csv
"""
import argparse
import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu

NON_FEATURE_COLS = ("patient_id", "label", "brugada_code")


def cohens_d(a, b):
    """Signed Cohen's d for group a vs. group b, pooled (unbiased) SD."""
    na, nb = len(a), len(b)
    if na < 2 or nb < 2:
        return np.nan
    pooled = np.sqrt(((na - 1) * a.std(ddof=1) ** 2 +
                      (nb - 1) * b.std(ddof=1) ** 2) / (na + nb - 2))
    if pooled == 0:
        return 0.0
    return (a.mean() - b.mean()) / pooled


def compute_effect_sizes(feat: pd.DataFrame) -> pd.DataFrame:
    features = [c for c in feat.columns if c not in NON_FEATURE_COLS]
    brs = feat[feat["label"] == 1]
    ctrl = feat[feat["label"] == 0]
    n_tests = len(features)

    rows = []
    for f in features:
        a = brs[f].dropna().values          # BrS
        b = ctrl[f].dropna().values         # control
        d = cohens_d(a, b)
        try:
            p = mannwhitneyu(a, b, alternative="two-sided").pvalue
        except ValueError:
            p = np.nan  # e.g. an all-identical feature
        rows.append({
            "feature": f,
            "cohens_d": d,
            "abs_d": abs(d),
            "mwu_p": p,
            "bonf_sig": bool(p < 0.05 / n_tests) if np.isfinite(p) else False,
        })
    return pd.DataFrame(rows)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", default="../data/features.csv")
    ap.add_argument("--output", default="../data/feature_effect_sizes.csv")
    ap.add_argument("--validate-against", default=None,
                    help="Optional path to the shipped feature_effect_sizes.csv "
                         "for a sanity-check correlation report.")
    args = ap.parse_args()

    feat = pd.read_csv(args.features)
    eff = compute_effect_sizes(feat)
    eff.to_csv(args.output, index=False)
    n_sig = int(eff["bonf_sig"].sum())
    print(f"Wrote {len(eff)} features to {args.output} "
          f"({n_sig} Bonferroni-significant).")

    if args.validate_against:
        orig = pd.read_csv(args.validate_against).set_index("feature")
        merged = eff.set_index("feature").join(orig, lsuffix="_new", rsuffix="_orig")
        corr = merged["cohens_d_new"].corr(merged["cohens_d_orig"])
        agree = (merged["bonf_sig_new"] == merged["bonf_sig_orig"]).mean()
        print(f"  cohens_d correlation vs. shipped = {corr:.4f}")
        print(f"  bonf_sig agreement vs. shipped   = {agree:.3f}")

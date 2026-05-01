# Data

This directory holds the Brugada-HUCA derived data files used throughout this repo.

- `records.csv` — per-patient inventory (label, BrS subtype, basal pattern, sudden-death flag)
- `brugada_raw.npz` — raw 12-lead ECG (X: [363,12,1200], 100Hz), pids, labels
- `preprocessed.npz` — band-pass filtered signal (Xf) + detected R-peaks + heart rate
- `features.csv` — 111-feature matrix + labels, output of `src/feature_extraction.py`
- `feature_effect_sizes.csv` — Cohen's d and Mann-Whitney U per feature, Bonferroni-flagged
- `cv_metrics.csv` — cross-validated AUROC/AUPRC/sensitivity/specificity for both models
- `error_analysis.csv` — per-patient FN/FP outcomes used in the concealed-phenotype analysis

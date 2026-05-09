# Data

This directory holds the Brugada-HUCA derived data files used throughout this repo.

- `records.csv` — per-patient inventory (label, BrS subtype, basal pattern, sudden-death flag)
- `brugada_raw.npz` — raw 12-lead ECG (X: [363,12,1200], 100Hz), pids, labels
- `preprocessed.npz` — band-pass filtered signal (Xf) + detected R-peaks + heart rate
- `features.csv` — 111-feature matrix + labels, output of `src/feature_extraction.py`
- `feature_effect_sizes.csv` — Cohen's d and Mann-Whitney U per feature, Bonferroni-flagged
- `cv_metrics.csv` — cross-validated AUROC/AUPRC/sensitivity/specificity for both models
- `error_analysis.csv` — per-patient FN/FP outcomes used in the concealed-phenotype analysis

## Column glossary (per-lead features)

Each of the 12 leads contributes 9 columns named `<LEAD>_<feature>`:

| Suffix | Meaning |
|---|---|
| `J_amp` | J-point (QRS-offset) amplitude, mV |
| `ST40` / `ST80` | ST-segment amplitude at J+40ms / J+80ms, mV |
| `ST_slope` | slope of the ST segment between +40ms and +80ms |
| `R_amp` / `S_amp` | R-wave / S-wave amplitude, mV |
| `QRS_dur` | QRS duration, ms |
| `T_amp` | signed T-wave amplitude, mV |
| `J_to_R_ratio` | J-point amplitude divided by R amplitude |

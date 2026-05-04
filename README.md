# Brugada Syndrome ECG Classifier

An interpretable machine-learning classifier for Brugada syndrome (BrS) on the Brugada-HUCA 12-lead ECG dataset — the code behind [this paper](paper/Brugada_Syndrome_ECG_Classifier_Paper.pdf).

> Independent research project. Not a diagnostic device — see Limitations.

## Motivation

Brugada syndrome is an inherited channelopathy that can cause sudden cardiac death in young, otherwise healthy people. Its diagnostic ECG signature is frequently subtle, intermittent, or absent on a resting ECG, and reliably unmasking it currently requires a provocative drug challenge that carries a small proarrhythmic risk. This project builds a transparent, feature-based classifier on the newly released Brugada-HUCA dataset and honestly evaluates where it succeeds and where it fails.

## Dataset

[Brugada-HUCA](https://physionet.org), PhysioNet v1.0.0 (Feb 2026) — 363 subjects (76 BrS, 287 controls), 12-lead ECG at 100 Hz, one record per patient.

## Repository structure

```
src/
  feature_extraction.py   raw/preprocessed ECG -> 111-feature table
  train_model.py          repeated stratified CV: XGBoost + Random Forest
  shap_explain.py         SHAP attribution on the final fitted model
  make_figures.py         regenerates every figure in the paper
data/                     feature tables, CV metrics, error analysis, raw checkpoints
figures/                  generated PNGs
paper/                    the manuscript PDF
```

## Method summary

1. Band-pass filter each lead (0.5-40Hz) and detect R-peaks.
2. Build a robust per-lead **median beat** (align on R-peak, take per-sample median).
3. Extract 9 features per lead (J-point, ST40/ST80, ST slope, R/S amplitude, QRS duration, T amplitude, J-to-R ratio) x 12 leads + 3 rhythm features = **111 features**.
4. Screen features with Cohen's *d* / Mann-Whitney U (Bonferroni-corrected) before modeling.
5. Train XGBoost (primary) and Random Forest (comparator) under **repeated stratified 5-fold CV, 10 repeats (50 folds)** — patient-level and leakage-free by construction.
6. Explain the fitted model with SHAP; tie errors to the `basal_pattern` clinical flag.

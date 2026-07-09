# Brugada Syndrome ECG Classifier

![Python](https://img.shields.io/badge/python-3.10%2B-blue) ![License](https://img.shields.io/badge/license-MIT-green)

An interpretable machine-learning classifier for Brugada syndrome (BrS) on the Brugada-HUCA 12-lead ECG dataset — the code behind [this paper](paper/Brugada_Syndrome_ECG_Classifier_Paper.pdf).

> Independent research project. Not a diagnostic device — see Limitations.

## Contents

- [Motivation](#motivation)
- [Dataset](#dataset)
- [Repository structure](#repository-structure)
- [Method summary](#method-summary)
- [Results](#results)
- [Error analysis](#error-analysis)
- [Reproducing this work](#reproducing-this-work)
- [Limitations](#limitations)
- [Citation](#citation)

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
4. Compute a **descriptive** univariate screen (Cohen's *d* / Mann-Whitney U, Bonferroni-corrected) with `compute_effect_sizes.py`. This is for interpretation and figures only — it is **not** feature selection: the models train on all 111 features, so nothing here can leak into the cross-validated numbers.
5. Train XGBoost (primary) and Random Forest (comparator) under **repeated stratified 5-fold CV, 10 repeats (50 folds)** — patient-level and leakage-free by construction (one ECG per patient).
6. Explain the model with SHAP — refit once on the **full** labeled set for interpretation only, kept separate from the held-out-fold performance above — and tie errors to the `basal_pattern` clinical flag.

## Results

| Metric | XGBoost | Random Forest |
|---|---|---|
| AUROC | 0.900 [0.80, 0.97] | 0.910 [0.84, 0.97] |
| AUPRC | 0.777 [0.57, 0.92] | 0.789 [0.64, 0.91] |
| Sensitivity @ 90% specificity | **0.763 [0.53, 0.94]** | 0.751 [0.53, 0.93] |

Brackets are 2.5–97.5 percentiles across the 50 CV folds. The sensitivity-at-90%-specificity interval is wide because it is an operating-point estimate on a small positive class — treat the point estimate accordingly.

SHAP attribution's top-3 features (V2 ST, V1 ST, V2 J-point amplitude) match the univariate effect-size screen almost exactly, and 5 of the top 10 SHAP features are in leads V1-V3. The SHAP model is refit on the full labeled set (standard for interpretation); it is not the source of the cross-validated numbers above, which come only from held-out folds.

## Error analysis

Out-of-fold sensitivity is **62% (33/53, 95% CI 49–74%)** when the patient's baseline ECG is non-pathological vs. **83% (19/23, 95% CI 63–93%)** when it is overtly abnormal (Wilson intervals; overall **68%, 52/76, 95% CI 57–78%**). The subgroups are small, so these intervals are wide and overlapping — read the split as directional, not as a precise effect size. Missed BrS cases sit at an intermediate V2 ST amplitude (0.174 mV) between controls (0.119 mV) and correctly caught cases (0.252 mV) — the model's failures concentrate on the same concealed phenotype that motivates sodium-channel-blocker provocation testing in clinical practice.

## Reproducing this work

```bash
pip install -r requirements.txt

cd src
# 0.5-40Hz filter + R-peak detection: brugada_raw.npz -> preprocessed.npz
python preprocess_raw.py --input ../data/brugada_raw.npz --output ../data/preprocessed_reproduced.npz --validate-against ../data/preprocessed.npz
python feature_extraction.py --input ../data/preprocessed.npz --output ../data/features_reproduced.csv
# descriptive univariate screen (interpretation/figures only, not feature selection)
python compute_effect_sizes.py --features ../data/features.csv --output ../data/feature_effect_sizes_reproduced.csv
python train_model.py --features ../data/features.csv --out ../data/cv_metrics_reproduced.csv
python shap_explain.py --features ../data/features.csv --out-dir ../data/
python make_figures.py --data-dir ../data --out-dir ../figures
```

**Checkpoints vs. raw download.** The pipeline starts from two provided checkpoints, `data/brugada_raw.npz` (raw signal) and `data/preprocessed.npz` (filtered + R-peaks). The upstream step that reads PhysioNet's raw WFDB `.dat`/`.hea` records and assembles `brugada_raw.npz` needs the PhysioNet download and is not scripted here; everything from `brugada_raw.npz` onward is reproducible with the commands above.

## Limitations

- Single-center, single-dataset — no external validation cohort.
- Small sample (76 BrS, only 7 type-2) — wide confidence intervals.
- 100 Hz sampling is coarse for fine J-point/QRS morphology.
- Labels are diagnostic, not per-beat — caps achievable sensitivity on a resting ECG alone.
- **This is a research baseline, not a validated diagnostic tool.**

## Citation

If you use this code or the accompanying paper, please cite the manuscript in `paper/` and the underlying [Brugada-HUCA](https://physionet.org) dataset (PhysioNet v1.0.0, Feb 2026).

## License

MIT — see [LICENSE](LICENSE).

## Contact

Sanjeet — sanjeet.jaysee@gmail.com

## Acknowledgments

Thanks to Hospital Universitario Central de Asturias for releasing the Brugada-HUCA dataset on PhysioNet, and to the authors of prior BrS-AI work cited in the paper for the context that made this project possible to situate honestly.

## Running tests

```bash
pip install -r requirements.txt
pytest tests/ -q
```

## Related work cited in the paper

Melo et al. 2023 (PNAS Nexus), Zanchi et al. 2023 (Europace), and Ronan et al. 2025 (Sci Rep) — see `paper/` for full citations. This repo is not a head-to-head comparison against those; it's an independent, interpretable feature-based baseline on the Brugada-HUCA dataset. Brugada-HUCA is also the ECG track of the International Data Science Challenge 2026, so other public baselines on this data exist — the aim here is transparency and clinical interpretability, not a novelty or leaderboard claim.

---

*Built as an independent passion project — see the paper for the full problem-to-model narrative.*

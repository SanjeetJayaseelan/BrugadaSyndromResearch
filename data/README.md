# Data

This directory holds the Brugada-HUCA derived data files used throughout this repo.

- `records.csv` — per-patient inventory (label, BrS subtype, basal pattern, sudden-death flag)
- `brugada_raw.npz` — raw 12-lead ECG (X: [363,12,1200], 100Hz), pids, labels
- `preprocessed.npz` — band-pass filtered signal (Xf) + detected R-peaks + heart rate

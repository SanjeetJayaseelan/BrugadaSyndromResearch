"""
feature_extraction.py

Reconstructs the 111-column interpretable feature table (features.csv) from the
preprocessed 12-lead ECG checkpoint (preprocessed.npz), following the median-beat
methodology described in the accompanying paper (Section 3.2).

This is an independent re-implementation written for reproducibility. See README
for a validation note on how closely it reproduces the original features.csv.
"""
import argparse
import numpy as np
import pandas as pd

FS = 100.0  # Hz


def median_beat(signal, rpeaks, pre_samples=30, post_samples=70):
    """Align all beats on their R-peak and take the per-sample median."""
    beats = []
    for r in rpeaks:
        lo, hi = r - pre_samples, r + post_samples
        if lo < 0 or hi > len(signal):
            continue
        beats.append(signal[lo:hi])
    if len(beats) < 2:
        return None
    return np.median(np.vstack(beats), axis=0)


def extract_lead_features(beat, pre_samples=30, fs=FS):
    """Extract the 9 interpretable features from one median beat.

    `beat` is indexed 0..len-1 with the R-peak at index `pre_samples`.
    """
    r_idx = pre_samples
    dt_ms = 1000.0 / fs

    # R amplitude: value at the R-peak sample itself
    r_amp = beat[r_idx]

    # S amplitude: minimum in a short window after the R-peak (within ~80 ms)
    s_win_end = min(len(beat), r_idx + int(round(80 / dt_ms)))
    s_amp = beat[r_idx:s_win_end].min() if s_win_end > r_idx else beat[r_idx]

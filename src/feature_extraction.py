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

    # QRS offset / J-point: first point after the R-peak, past the S-wave trough,
    # where the beat's slope flattens (below a small threshold) for a sustained span.
    deriv = np.gradient(beat)
    search_start = int(np.argmin(beat[r_idx:s_win_end])) + r_idx if s_win_end > r_idx else r_idx
    j_idx = None
    flat_thresh = 0.02 * (np.max(beat) - np.min(beat) + 1e-9)
    for i in range(search_start, min(len(beat) - 3, search_start + int(round(120 / dt_ms)))):
        if np.all(np.abs(deriv[i:i + 3]) < flat_thresh):
            j_idx = i
            break
    if j_idx is None:
        j_idx = min(len(beat) - 1, search_start + int(round(60 / dt_ms)))

    j_amp = beat[j_idx]

    # QRS onset: last point before the R-peak where the beat departs from baseline
    q_idx = r_idx
    baseline = np.median(beat[:max(1, pre_samples - 15)])
    dep_thresh = 0.05 * (np.max(beat) - np.min(beat) + 1e-9)
    for i in range(r_idx, 0, -1):
        if abs(beat[i] - baseline) < dep_thresh:
            q_idx = i
            break
    qrs_dur = (j_idx - q_idx) * dt_ms

    # ST amplitudes at J+40ms / J+80ms
    def amp_at_offset(offset_ms):
        idx = j_idx + int(round(offset_ms / dt_ms))
        idx = min(idx, len(beat) - 1)
        return beat[idx]

    st40 = amp_at_offset(40)
    st80 = amp_at_offset(80)
    st_slope = (st80 - st40) / 40.0  # mV/ms over the 40-80ms window

    # T-wave amplitude: signed peak-magnitude in a search window ~120-320ms post J-point
    t_start = j_idx + int(round(60 / dt_ms))
    t_end = min(len(beat), j_idx + int(round(280 / dt_ms)))
    if t_end > t_start:
        seg = beat[t_start:t_end]
        t_amp = seg[np.argmax(np.abs(seg))]
    else:
        t_amp = np.nan

    j_to_r_ratio = j_amp / r_amp if abs(r_amp) > 1e-6 else np.nan

    return {
        "J_amp": j_amp, "ST40": st40, "ST80": st80, "ST_slope": st_slope,
        "R_amp": r_amp, "S_amp": s_amp, "QRS_dur": qrs_dur, "T_amp": t_amp,
        "J_to_R_ratio": j_to_r_ratio,
    }

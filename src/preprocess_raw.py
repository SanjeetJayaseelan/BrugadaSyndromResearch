"""
preprocess_raw.py

Turns the raw 12-lead ECG checkpoint (brugada_raw.npz) into the preprocessed
checkpoint (preprocessed.npz) that feature_extraction.py consumes, implementing
the two signal-processing steps the paper's Method summary opens with:

  1. Band-pass filter each lead at 0.5-40 Hz (4th-order Butterworth, zero-phase).
  2. Detect R-peaks on lead II (QRS-band energy + peak picking) and derive HR.

  brugada_raw.npz : X (n,12,1200) raw signal + pids/y/brug3/basal/sudden/leads
  preprocessed.npz: Xf (filtered) + rpeaks + hr + the same metadata

NOTE ON THE FULL PIPELINE. PhysioNet ships Brugada-HUCA as WFDB .dat/.hea
records. The step that reads those raw records and assembles brugada_raw.npz
requires the PhysioNet download (and the `wfdb` package) and is therefore NOT
scripted in this repo; brugada_raw.npz is provided as a checkpoint. This script
covers the filtering + R-peak stage, which is fully reproducible from the
shipped checkpoint. Like feature_extraction.py, it is an independent
re-implementation — run with --validate-against to see how closely it
reproduces the shipped preprocessed.npz.

Usage:
    python preprocess_raw.py --input ../data/brugada_raw.npz \
        --output ../data/preprocessed_reproduced.npz \
        --validate-against ../data/preprocessed.npz
"""
import argparse
import numpy as np
from scipy.signal import butter, filtfilt, find_peaks

FS = 100.0  # Hz
CARRY_KEYS = ("pids", "y", "brug3", "basal", "sudden", "leads")


def bandpass(sig, lo=0.5, hi=40.0, fs=FS, order=4):
    """Zero-phase Butterworth band-pass for one lead."""
    ny = fs / 2.0
    b, a = butter(order, [lo / ny, min(hi / ny, 0.99)], btype="band")
    return filtfilt(b, a, sig)


def detect_rpeaks(lead_ii, fs=FS):
    """R-peaks from lead II: isolate the QRS band, square to an energy signal,
    then pick peaks above a mean+SD threshold with a 300 ms refractory gap."""
    qrs = bandpass(lead_ii, 5.0, 25.0, fs)
    energy = qrs ** 2
    thresh = energy.mean() + 0.5 * energy.std()
    peaks, _ = find_peaks(energy, height=thresh, distance=int(0.3 * fs))
    return peaks.astype(int)


def heart_rate(rpeaks, fs=FS):
    """Median-RR heart rate in bpm; NaN if fewer than two beats."""
    if len(rpeaks) < 2:
        return np.nan
    rr_s = np.diff(rpeaks) / fs
    return float(60.0 / np.median(rr_s))


def preprocess(npz_path):
    raw = np.load(npz_path, allow_pickle=True)
    X = raw["X"]                       # (n, 12, T) raw
    leads = list(raw["leads"])
    ii = leads.index("II")

    n, n_leads, T = X.shape
    Xf = np.empty_like(X, dtype=np.float32)
    rpeaks = np.empty(n, dtype=object)
    hr = np.empty(n, dtype=np.float64)

    for s in range(n):
        for li in range(n_leads):
            Xf[s, li, :] = bandpass(X[s, li, :])
        rp = detect_rpeaks(Xf[s, ii, :])
        rpeaks[s] = rp
        hr[s] = heart_rate(rp)

    out = {"Xf": Xf, "rpeaks": rpeaks, "hr": hr}
    for k in CARRY_KEYS:
        if k in raw:
            out[k] = raw[k]
    return out


def validate(out, ref_path):
    ref = np.load(ref_path, allow_pickle=True)
    Xf_ref = ref["Xf"]
    corr = []
    for s in range(0, out["Xf"].shape[0], 10):
        for li in range(out["Xf"].shape[1]):
            corr.append(np.corrcoef(out["Xf"][s, li], Xf_ref[s, li])[0, 1])
    print(f"  filtered-signal corr vs. shipped Xf: median={np.median(corr):.4f}")

    if "rpeaks" in ref:
        rp_ref = ref["rpeaks"]
        prec, rec = [], []
        for s in range(out["Xf"].shape[0]):
            mine = out["rpeaks"][s]
            orig = np.asarray(rp_ref[s]).astype(int)
            matched = sum(any(abs(int(m) - int(o)) <= 3 for o in orig) for m in mine)
            prec.append(matched / max(len(mine), 1))
            rec.append(matched / max(len(orig), 1))
        print(f"  R-peak match vs. shipped (+/-3 samples): "
              f"precision={np.mean(prec):.2f} recall={np.mean(rec):.2f}")
    if "hr" in ref:
        hr_corr = np.corrcoef(np.nan_to_num(out["hr"]),
                              np.nan_to_num(ref["hr"]))[0, 1]
        print(f"  HR correlation vs. shipped: {hr_corr:.3f}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default="../data/brugada_raw.npz")
    ap.add_argument("--output", default="../data/preprocessed_reproduced.npz")
    ap.add_argument("--validate-against", default=None,
                    help="Optional path to the shipped preprocessed.npz for a "
                         "reproduction report (Xf corr, R-peak match, HR corr).")
    args = ap.parse_args()

    out = preprocess(args.input)
    np.savez_compressed(args.output, **out)
    print(f"Wrote {out['Xf'].shape[0]} filtered records + R-peaks to {args.output}")

    if args.validate_against:
        validate(out, args.validate_against)

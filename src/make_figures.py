"""
make_figures.py

Regenerates every figure used in the paper directly from the raw / preprocessed
ECG checkpoints and the feature / error-analysis tables:

  fig1  Example right-precordial waveforms (BrS type-1 vs control)
  fig2  Preprocessing QC (raw vs filtered, R-peak detection)
  fig3  Top-6 discriminating feature distributions
  fig4  Error analysis (subgroup sensitivity, V2 ST by outcome group)
  fig5  ROC / PR / calibration curves (50-fold repeated stratified CV)
  fig6  SHAP attribution + confusion matrix

Usage:
    python make_figures.py --data-dir ../data --out-dir ../figures
"""
import argparse
import pickle
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt




def fig1_waveforms(data_dir, out_dir, brs_pid=188981, ctrl_pid=251972):
    raw = np.load(f"{data_dir}/brugada_raw.npz", allow_pickle=True)
    leads = list(raw["leads"]); pids = raw["pids"]; X = raw["X"]
    fs = 100.0
    t = np.arange(X.shape[2]) / fs
    brs_idx = int(np.where(pids == brs_pid)[0][0])
    ctl_idx = int(np.where(pids == ctrl_pid)[0][0])
    show_leads = ["V1", "V2", "V3", "II"]

    fig, axs = plt.subplots(len(show_leads), 2, figsize=(11, 9), sharex=True)
    fig.suptitle("Right-precordial leads: Brugada type-1 vs control", fontsize=13)
    for r, ld in enumerate(show_leads):
        li = leads.index(ld)
        axs[r, 0].plot(t, X[brs_idx, li, :], color="#c1440e", lw=1)
        axs[r, 1].plot(t, X[ctl_idx, li, :], color="#33475b", lw=1)
        for c in (0, 1):
            axs[r, c].axhline(0, color="gray", lw=0.6)
            axs[r, c].spines["top"].set_visible(False)
            axs[r, c].spines["right"].set_visible(False)
        axs[r, 0].set_ylabel(ld)
    axs[0, 0].set_title(f"BrS type-1 (pid {brs_pid})")
    axs[0, 1].set_title(f"Control (pid {ctrl_pid})")
    axs[-1, 0].set_xlabel("Time (s)"); axs[-1, 1].set_xlabel("Time (s)")
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(f"{out_dir}/fig1_example_waveforms.png", dpi=180)
    plt.close(fig)

def fig2_preprocessing_qc(data_dir, out_dir, brs_pid=188981, ctrl_pid=251972):
    raw = np.load(f"{data_dir}/brugada_raw.npz", allow_pickle=True)
    pre = np.load(f"{data_dir}/preprocessed.npz", allow_pickle=True)
    leads = list(raw["leads"]); pids = raw["pids"]; fs = 100.0
    X, Xf, rpeaks = raw["X"], pre["Xf"], pre["rpeaks"]
    t = np.arange(X.shape[2]) / fs
    brs_idx = int(np.where(pids == brs_pid)[0][0])
    ctl_idx = int(np.where(pids == ctrl_pid)[0][0])
    li_v2, li_ii = leads.index("V2"), leads.index("II")

    fig, axs = plt.subplots(3, 2, figsize=(12, 9.5))
    fig.suptitle("Preprocessing QC: bandpass filtering and R-peak detection", fontsize=13)
    for c, (idx, title) in enumerate([(brs_idx, f"BrS (pid {brs_pid}) — lead V2"),
                                       (ctl_idx, f"Control (pid {ctrl_pid}) — lead V2")]):
        axs[0, c].plot(t, X[idx, li_v2, :], color="gray", lw=0.9, label="raw")
        axs[0, c].plot(t, Xf[idx, li_v2, :], color="#c1440e", lw=1.1, label="filtered")
        axs[0, c].set_title(title); axs[0, c].set_ylabel("V2 (mV)")
        if c == 0:
            axs[0, c].legend(loc="upper right", frameon=False, fontsize=8)

        removed = X[idx, li_v2, :] - Xf[idx, li_v2, :]
        axs[1, c].plot(t, removed, color="#2b6cb0", lw=1)
        axs[1, c].set_ylabel("removed\n(baseline+HF)")

        sig = Xf[idx, li_ii, :]
        axs[2, c].plot(t, sig, color="#33475b", lw=0.9)
        rp = np.asarray(rpeaks[idx]).astype(int)
        rp = rp[(rp >= 0) & (rp < len(sig))]
        axs[2, c].scatter(t[rp], sig[rp], marker="v", color="#e07b00", s=45, zorder=5)
        axs[2, c].set_ylabel("II + R-peaks"); axs[2, c].set_xlabel("Time (s)")

        for r in range(3):
            axs[r, c].spines["top"].set_visible(False)
            axs[r, c].spines["right"].set_visible(False)

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(f"{out_dir}/fig2_preprocessing_qc.png", dpi=180)
    plt.close(fig)

def fig3_feature_distributions(data_dir, out_dir):
    feat = pd.read_csv(f"{data_dir}/features.csv")
    eff = pd.read_csv(f"{data_dir}/feature_effect_sizes.csv").set_index("feature")
    top6 = ["V2_ST40", "V1_ST40", "V4_QRS_dur", "V3_QRS_dur", "V2_QRS_dur", "V2_T_amp"]
    units = {"V2_ST40": "mV", "V1_ST40": "mV", "V4_QRS_dur": "ms", "V3_QRS_dur": "ms",
             "V2_QRS_dur": "ms", "V2_T_amp": "mV"}
    titles = {"V2_ST40": "V2 ST (J+40ms)", "V1_ST40": "V1 ST (J+40ms)",
              "V4_QRS_dur": "V4 QRS duration", "V3_QRS_dur": "V3 QRS duration",
              "V2_QRS_dur": "V2 QRS duration", "V2_T_amp": "V2 T-wave amplitude"}
    ctrl, brs = feat[feat.label == 0], feat[feat.label == 1]
    rng = np.random.default_rng(7)

    fig, axs = plt.subplots(2, 3, figsize=(13, 8.5))
    fig.suptitle("Top discriminating features: BrS vs control (right-precordial signature)", fontsize=13)
    axs = axs.ravel()
    for i, f in enumerate(top6):
        ax = axs[i]; d = eff.loc[f, "cohens_d"]
        data = [ctrl[f].dropna().values, brs[f].dropna().values]
        parts = ax.violinplot(data, showextrema=True)
        for pc, col in zip(parts["bodies"], ["#8fa3b3", "#e5a29a"]):
            pc.set_facecolor(col); pc.set_alpha(0.55); pc.set_edgecolor("none")
        for key in ("cbars", "cmins", "cmaxes"):
            parts[key].set_color("gray"); parts[key].set_linewidth(1)
        for gi, arr in enumerate(data, start=1):
            q1, med, q3 = np.percentile(arr, [25, 50, 75])
            ax.hlines([q1, med, q3], gi - 0.18, gi + 0.18, color="dimgray", lw=1)
        for gi, (arr, col) in enumerate(zip(data, ["#33475b", "#c1440e"]), start=1):
            x = gi + rng.normal(0, 0.045, size=len(arr))
            ax.scatter(x, arr, s=6, color=col, alpha=0.45, linewidths=0)
        ax.set_xticks([1, 2]); ax.set_xticklabels(["Control", "BrS"])
        ax.set_title(f"{titles[f]}  (d={'+' if d >= 0 else ''}{d:.2f})", fontsize=10.5)
        ax.set_ylabel(units[f])
        ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(f"{out_dir}/fig3_feature_distributions.png", dpi=180)
    plt.close(fig)
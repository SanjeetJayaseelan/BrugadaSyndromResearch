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
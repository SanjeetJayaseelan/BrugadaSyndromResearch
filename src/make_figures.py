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

# consistent spine styling across all panels (top/right hidden for a cleaner look)




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

def fig4_error_analysis(data_dir, out_dir):
    feat = pd.read_csv(f"{data_dir}/features.csv")
    recs = pd.read_csv(f"{data_dir}/records.csv")[["patient_id", "basal_pattern", "brs_type"]]
    err = pd.read_csv(f"{data_dir}/error_analysis.csv")
    fn_ids, fp_ids = set(err.loc[err.error == "FN", "patient_id"]), set(err.loc[err.error == "FP", "patient_id"])
    feat = feat.merge(recs, on="patient_id", how="left")

    def outcome(row):
        if row.label == 1:
            return "FN" if row.patient_id in fn_ids else "TP"
        return "FP" if row.patient_id in fp_ids else "TN"
    feat["outcome"] = feat.apply(outcome, axis=1)
    brs = feat[feat.label == 1].copy()

    def sens(df):
        return (df.outcome == "TP").sum(), len(df)

    groups = [("Concealed\nbaseline", *sens(brs[brs.basal_pattern == 0])),
              ("Overt\nbaseline", *sens(brs[brs.basal_pattern == 1])),
              ("Type-1", *sens(brs[brs.brs_type == "type1"])),
              ("Type-2", *sens(brs[brs.brs_type == "type2"]))]
    overall_sens = 100 * (brs.outcome == "TP").sum() / len(brs)

    fig, axs = plt.subplots(1, 2, figsize=(13, 5.5))
    colors = ["#c1440e", "#e07b00", "#7f8c8d", "#7f8c8d"]
    labels = [g[0] for g in groups]; vals = [100 * g[1] / g[2] for g in groups]
    bars = axs[0].bar(labels, vals, color=colors, width=0.62)
    for b, g in zip(bars, groups):
        axs[0].text(b.get_x() + b.get_width() / 2, b.get_height() + 1.5, f"{g[1]}/{g[2]}", ha="center", fontsize=10)
    axs[0].axhline(overall_sens, color="gray", ls="--", lw=1)
    axs[0].set_ylabel("Sensitivity (%)"); axs[0].set_ylim(0, 100)
    axs[0].set_title("BrS detection by subgroup — concealed cases missed most")
    axs[0].spines["top"].set_visible(False); axs[0].spines["right"].set_visible(False)

    data = [feat.loc[feat.outcome == "TN", "V2_ST40"].values,
            feat.loc[feat.outcome == "FN", "V2_ST40"].values,
            feat.loc[feat.outcome == "TP", "V2_ST40"].values]
    rng = np.random.default_rng(3)
    parts = axs[1].violinplot(data, showextrema=True)
    for pc, col in zip(parts["bodies"], ["#8fa3b3", "#e5a29a", "#9fd0ae"]):
        pc.set_facecolor(col); pc.set_alpha(0.55); pc.set_edgecolor("none")
    for key in ("cbars", "cmins", "cmaxes"):
        parts[key].set_color("gray"); parts[key].set_linewidth(1)
    for gi, arr in enumerate(data, start=1):
        q1, med, q3 = np.percentile(arr, [25, 50, 75])
        axs[1].hlines([q1, med, q3], gi - 0.18, gi + 0.18, color="dimgray", lw=1)
    for gi, (arr, col) in enumerate(zip(data, ["#33475b", "#c1440e", "#1a7a3c"]), start=1):
        x = gi + rng.normal(0, 0.05, size=len(arr))
        axs[1].scatter(x, arr, s=8, color=col, alpha=0.5, linewidths=0)
    axs[1].set_xticks([1, 2, 3]); axs[1].set_xticklabels(["Controls", "Missed BrS\n(FN)", "Caught BrS\n(TP)"])
    axs[1].set_ylabel("V2 ST amplitude (mV)")
    axs[1].set_title("Missed BrS have intermediate V2 ST elevation")
    axs[1].spines["top"].set_visible(False); axs[1].spines["right"].set_visible(False)
    plt.tight_layout()
    plt.savefig(f"{out_dir}/fig4_error_analysis.png", dpi=180)
    plt.close(fig)

def fig5_roc_pr_calibration(data_dir, out_dir):
    from sklearn.model_selection import RepeatedStratifiedKFold
    from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score
    from sklearn.calibration import calibration_curve
    import xgboost as xgb

    feat = pd.read_csv(f"{data_dir}/features.csv")
    X = feat.drop(columns=["patient_id", "label", "brugada_code"]).values
    y = feat["label"].values
    spw = (len(y) - y.sum()) / y.sum()

    rskf = RepeatedStratifiedKFold(n_splits=5, n_repeats=10, random_state=42)
    mean_fpr, mean_rec = np.linspace(0, 1, 200), np.linspace(0, 1, 200)
    fold_tprs, fold_precs, fold_aucs, fold_aps = [], [], [], []
    oof_sum, oof_cnt = np.zeros(len(y)), np.zeros(len(y))

    for tr, te in rskf.split(X, y):
        model = xgb.XGBClassifier(n_estimators=300, max_depth=3, learning_rate=0.05,
                                   scale_pos_weight=spw, eval_metric="logloss",
                                   subsample=0.9, colsample_bytree=0.9, random_state=0, n_jobs=2)
        model.fit(X[tr], y[tr])
        p = model.predict_proba(X[te])[:, 1]
        oof_sum[te] += p; oof_cnt[te] += 1
        fpr, tpr, _ = roc_curve(y[te], p)
        fold_aucs.append(auc(fpr, tpr)); fold_tprs.append(np.interp(mean_fpr, fpr, tpr))
        prec, rec, _ = precision_recall_curve(y[te], p)
        fold_aps.append(average_precision_score(y[te], p))
        order = np.argsort(rec)
        fold_precs.append(np.interp(mean_rec, rec[order], prec[order]))

    oof = oof_sum / np.maximum(oof_cnt, 1)
    fold_tprs, fold_precs = np.array(fold_tprs), np.array(fold_precs)
    mean_tpr = fold_tprs.mean(0); lo_tpr, hi_tpr = np.percentile(fold_tprs, [2.5, 97.5], axis=0)
    mean_prec = fold_precs.mean(0); lo_prec, hi_prec = np.percentile(fold_precs, [2.5, 97.5], axis=0)
    mean_auc, mean_ap = np.mean(fold_aucs), np.mean(fold_aps)
    frac_pos, mean_pred = calibration_curve(y, oof, n_bins=8, strategy="quantile")

    fig, axs = plt.subplots(1, 3, figsize=(15, 4.6))
    axs[0].plot(mean_fpr, mean_tpr, color="#c1440e", lw=2, label=f"XGBoost (AUROC {mean_auc:.2f})")
    axs[0].fill_between(mean_fpr, lo_tpr, hi_tpr, color="#c1440e", alpha=0.15)
    axs[0].plot([0, 1], [0, 1], color="gray", ls="--", lw=1)
    axs[0].set_xlabel("1 − Specificity"); axs[0].set_ylabel("Sensitivity")
    axs[0].set_title("ROC (50 folds, 95% band)"); axs[0].legend(loc="lower right", frameon=False, fontsize=9)

    prevalence = y.mean()
    axs[1].plot(mean_rec, mean_prec, color="#c1440e", lw=2, label=f"XGBoost (AUPRC {mean_ap:.2f})")
    axs[1].fill_between(mean_rec, lo_prec, hi_prec, color="#c1440e", alpha=0.15)
    axs[1].axhline(prevalence, color="gray", ls="--", lw=1, label=f"prevalence ({prevalence:.2f})")
    axs[1].set_xlabel("Recall (Sensitivity)"); axs[1].set_ylabel("Precision")
    axs[1].set_title("Precision–Recall"); axs[1].legend(loc="upper right", frameon=False, fontsize=9)

    axs[2].plot(mean_pred, frac_pos, marker="o", color="#c1440e", label="XGBoost")
    axs[2].plot([0, 1], [0, 1], color="gray", ls="--", lw=1)
    axs[2].set_xlabel("Predicted probability"); axs[2].set_ylabel("Observed frequency")
    axs[2].set_title("Calibration (out-of-fold)"); axs[2].legend(loc="upper left", frameon=False, fontsize=9)

    for ax in axs:
        ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    plt.tight_layout()
    plt.savefig(f"{out_dir}/fig5_roc_pr_calibration.png", dpi=180)
    plt.close(fig)
    return mean_auc, mean_ap

def fig6_shap_confusion(data_dir, out_dir, confusion=(272, 15, 24, 52)):
    """confusion = (TN, FP, FN, TP), taken from data/error_analysis.csv."""
    import xgboost as xgb
    import shap

    feat = pd.read_csv(f"{data_dir}/features.csv")
    X = feat.drop(columns=["patient_id", "label", "brugada_code"])
    y = feat["label"].values
    feature_names = X.columns.tolist()
    Xv = X.values
    spw = (len(y) - y.sum()) / y.sum()

    model = xgb.XGBClassifier(n_estimators=300, max_depth=3, learning_rate=0.05,
                               scale_pos_weight=spw, eval_metric="logloss",
                               subsample=0.9, colsample_bytree=0.9, random_state=0, n_jobs=2)
    model.fit(Xv, y)
    sv = np.array(shap.TreeExplainer(model).shap_values(Xv))
    if sv.ndim == 3:
        sv = sv[:, :, 1]
    mean_abs = np.abs(sv).mean(axis=0)
    order = np.argsort(mean_abs)[::-1][:10]
    red_leads = ("V1_", "V2_", "V3_")

    fig = plt.figure(figsize=(15, 5.2))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.3, 1, 1])
    ax0 = fig.add_subplot(gs[0])
    rng = np.random.default_rng(1)
    for row, i in enumerate(order[::-1]):
        vals, fvals = sv[:, i], Xv[:, i]
        ranks = np.argsort(np.argsort(fvals)) / max(len(fvals) - 1, 1)
        y_jitter = row + rng.normal(0, 0.12, size=len(vals))
        ax0.scatter(vals, y_jitter, c=ranks, cmap="cool", s=9, alpha=0.75, linewidths=0)
    ax0.axvline(0, color="gray", lw=0.8)
    ax0.set_yticks(range(len(order))); ax0.set_yticklabels([feature_names[i] for i in order[::-1]])
    ax0.set_xlabel("SHAP value (impact on model output)"); ax0.set_title("SHAP feature attribution (XGBoost)")

    ax1 = fig.add_subplot(gs[1])
    colors = ["#c1440e" if any(feature_names[i].startswith(p) for p in red_leads) else "#8fa3b3" for i in order]
    ax1.barh(range(len(order)), [mean_abs[i] for i in order][::-1], color=colors[::-1])
    ax1.set_yticks(range(len(order))); ax1.set_yticklabels([feature_names[i] for i in order[::-1]])
    ax1.set_xlabel("mean |SHAP|"); ax1.set_title("Top-10 importance (red = V1–V3)")

    ax2 = fig.add_subplot(gs[2])
    tn, fp, fn, tp = confusion
    cm = np.array([[tn, fp], [fn, tp]])
    ax2.imshow(cm, cmap="Blues")
    ax2.set_xticks([0, 1]); ax2.set_xticklabels(["Control", "BrS"])
    ax2.set_yticks([0, 1]); ax2.set_yticklabels(["Control", "BrS"])
    ax2.set_xlabel("Predicted"); ax2.set_ylabel("True")
    ax2.set_title("Confusion matrix (out-of-fold, thr=0.5)")
    for i in range(2):
        for j in range(2):
            ax2.text(j, i, str(cm[i, j]), ha="center", va="center",
                      color="white" if cm[i, j] > 150 else "black", fontsize=13)

    for ax in (ax0, ax1):
        ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    plt.tight_layout()
    plt.savefig(f"{out_dir}/fig6_shap_confusion.png", dpi=180)
    plt.close(fig)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="../data")
    ap.add_argument("--out-dir", default="../figures")
    args = ap.parse_args()

    fig1_waveforms(args.data_dir, args.out_dir)
    print("fig1 done")
    fig2_preprocessing_qc(args.data_dir, args.out_dir)
    print("fig2 done")
    fig3_feature_distributions(args.data_dir, args.out_dir)
    print("fig3 done")
    fig4_error_analysis(args.data_dir, args.out_dir)
    print("fig4 done")
    fig5_roc_pr_calibration(args.data_dir, args.out_dir)
    print("fig5 done")
    fig6_shap_confusion(args.data_dir, args.out_dir)
    print("fig6 done")

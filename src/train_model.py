"""
train_model.py

Trains and cross-validates the BrS classifier described in the paper:
  - XGBoost (primary): 300 trees, max_depth=3, lr=0.05, scale_pos_weight for imbalance
  - Random Forest (comparator): class_weight='balanced'
  - Validation: repeated stratified 5-fold CV, 10 repeats (50 folds total)

Because Brugada-HUCA has exactly one ECG per patient, every CV split here is
automatically patient-level and leakage-free.

Usage:
    python train_model.py --features ../data/features.csv --out ../data/cv_metrics_reproduced.csv
"""
import argparse
import numpy as np
import pandas as pd
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb


def run_cv(X: 'np.ndarray', y: 'np.ndarray', model_name: str,
           n_splits: int = 5, n_repeats: int = 10, seed: int = 42):
    rskf = RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=seed)
    n_pos, n_neg = y.sum(), len(y) - y.sum()
    spw = n_neg / n_pos

    aurocs, auprcs, senss, specs, sens90, accs = [], [], [], [], [], []
    oof_sum, oof_cnt = np.zeros(len(y)), np.zeros(len(y))

    for tr, te in rskf.split(X, y):
        if model_name == "xgboost":
            model = xgb.XGBClassifier(n_estimators=300, max_depth=3, learning_rate=0.05,
                                       scale_pos_weight=spw, eval_metric="logloss",
                                       subsample=0.9, colsample_bytree=0.9,
                                       random_state=0, n_jobs=2)
        elif model_name == "random_forest":
            model = RandomForestClassifier(n_estimators=400, max_depth=None,
                                            class_weight="balanced", random_state=0, n_jobs=2)
        else:
            raise ValueError(model_name)

        model.fit(X[tr], y[tr])
        p = model.predict_proba(X[te])[:, 1]
        oof_sum[te] += p
        oof_cnt[te] += 1

        fpr, tpr, _ = roc_curve(y[te], p)
        aurocs.append(auc(fpr, tpr))
        auprcs.append(average_precision_score(y[te], p))

        pred = (p >= 0.5).astype(int)
        tp = ((pred == 1) & (y[te] == 1)).sum()
        fn = ((pred == 0) & (y[te] == 1)).sum()
        tn = ((pred == 0) & (y[te] == 0)).sum()
        fp = ((pred == 1) & (y[te] == 0)).sum()
        senss.append(tp / (tp + fn) if (tp + fn) else np.nan)
        specs.append(tn / (tn + fp) if (tn + fp) else np.nan)
        accs.append((tp + tn) / len(te))

        # sensitivity at fixed 90% specificity
        thresholds = np.linspace(0, 1, 500)
        best = 0.0
        for t in thresholds:
            pr = (p >= t).astype(int)
            tn_ = ((pr == 0) & (y[te] == 0)).sum()
            fp_ = ((pr == 1) & (y[te] == 0)).sum()
            spec_t = tn_ / (tn_ + fp_) if (tn_ + fp_) else 0
            if spec_t >= 0.90:
                tp_ = ((pr == 1) & (y[te] == 1)).sum()
                fn_ = ((pr == 0) & (y[te] == 1)).sum()
                sens_t = tp_ / (tp_ + fn_) if (tp_ + fn_) else 0
                best = max(best, sens_t)
        sens90.append(best)

    def summarize(arr):
        arr = np.array(arr)
        return arr.mean(), arr.std(), np.percentile(arr, 2.5), np.percentile(arr, 97.5)

    metrics = {}
    for name, arr in [("auroc", aurocs), ("auprc", auprcs), ("sens", senss),
                       ("spec", specs), ("sens_at_90spec", sens90), ("acc", accs)]:
        mean, std, lo, hi = summarize(arr)
        metrics[name] = {"mean": mean, "std": std, "ci_lo": lo, "ci_hi": hi}

    return metrics, oof_sum / np.maximum(oof_cnt, 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", default="../data/features.csv")
    ap.add_argument("--out", default="../data/cv_metrics_reproduced.csv")
    args = ap.parse_args()

    feat = pd.read_csv(args.features)
    X = feat.drop(columns=["patient_id", "label", "brugada_code"], errors="ignore").values
    y = feat["label"].values

    rows = []
    for model_name in ["xgboost", "random_forest"]:
        print(f"Running repeated stratified 5x10-fold CV for {model_name}...")
        metrics, oof = run_cv(X, y, model_name)
        for stat in ["mean", "std", "ci_lo", "ci_hi"]:
            rows.append({"model": model_name, "stat": stat,
                          **{m: metrics[m][stat] for m in metrics}})
        print(f"  AUROC = {metrics['auroc']['mean']:.3f} "
              f"[{metrics['auroc']['ci_lo']:.2f}, {metrics['auroc']['ci_hi']:.2f}]")

    out_df = pd.DataFrame(rows)
    out_df.to_csv(args.out, index=False)
    print(f"Wrote cross-validated metrics to {args.out}")


if __name__ == "__main__":
    main()

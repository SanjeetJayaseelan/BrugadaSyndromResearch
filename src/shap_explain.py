"""
shap_explain.py

Fits the primary XGBoost model on the full feature table and computes SHAP
(SHapley Additive exPlanations) attribution, to check that the model relies on
the clinically expected right-precordial ST/J-point signal rather than an
incidental artifact.

Usage:
    python shap_explain.py --features ../data/features.csv --out-dir ../data/
"""
import argparse
import numpy as np
import pandas as pd
import xgboost as xgb
import shap


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", default="../data/features.csv")
    ap.add_argument("--out-dir", default="../data/")
    ap.add_argument("--top-n", type=int, default=10)
    args = ap.parse_args()

    feat = pd.read_csv(args.features)
    X = feat.drop(columns=["patient_id", "label", "brugada_code"], errors="ignore")
    y = feat["label"].values
    feature_names = X.columns.tolist()
    Xv = X.values

    n_pos, n_neg = y.sum(), len(y) - y.sum()
    model = xgb.XGBClassifier(n_estimators=300, max_depth=3, learning_rate=0.05,
                               scale_pos_weight=n_neg / n_pos, eval_metric="logloss",
                               subsample=0.9, colsample_bytree=0.9, random_state=0, n_jobs=2)
    model.fit(Xv, y)

    explainer = shap.TreeExplainer(model)
    shap_values = np.array(explainer.shap_values(Xv))
    if shap_values.ndim == 3:  # some shap/xgboost version combos return (n, features, classes)
        shap_values = shap_values[:, :, 1]

    mean_abs = np.abs(shap_values).mean(axis=0)
    order = np.argsort(mean_abs)[::-1][: args.top_n]
    top = pd.DataFrame({
        "feature": [feature_names[i] for i in order],
        "mean_abs_shap": [mean_abs[i] for i in order],
    })
    print("Top features by mean |SHAP|:")
    print(top.to_string(index=False))

    np.save(args.out_dir + "shap_values.npy", shap_values)
    top.to_csv(args.out_dir + "shap_top_features.csv", index=False)


if __name__ == "__main__":
    main()

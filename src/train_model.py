"""
train_model.py

Trains and cross-validates the BrS classifier: XGBoost (primary) and Random Forest
(comparator), validated with repeated stratified 5-fold CV (10 repeats, 50 folds).

Because Brugada-HUCA has exactly one ECG per patient, every CV split here is
automatically patient-level and leakage-free.
"""
import argparse
import numpy as np
import pandas as pd
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb


def run_cv(X, y, model_name, n_splits=5, n_repeats=10, seed=42):
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

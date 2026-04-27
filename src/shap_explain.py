"""
shap_explain.py

Fits the primary XGBoost model on the full feature table and computes SHAP
attribution, to check the model relies on the clinically expected
right-precordial ST/J-point signal rather than an incidental artifact.
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

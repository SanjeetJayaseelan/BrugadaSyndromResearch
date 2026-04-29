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



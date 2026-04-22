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

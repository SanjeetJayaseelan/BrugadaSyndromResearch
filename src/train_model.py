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

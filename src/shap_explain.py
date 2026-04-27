"""
shap_explain.py

Fits the primary XGBoost model on the full feature table and computes SHAP
attribution, to check the model relies on the clinically expected
right-precordial ST/J-point signal rather than an incidental artifact.
"""
import argparse
import numpy as np
import pandas as pd

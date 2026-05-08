"""Smoke tests for median_beat() and extract_lead_features()."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
import numpy as np
from feature_extraction import median_beat, extract_lead_features


def test_median_beat_shape():
    signal = np.sin(np.linspace(0, 20 * np.pi, 1200))
    rpeaks = np.arange(60, 1140, 100)
    beat = median_beat(signal, rpeaks, pre_samples=30, post_samples=70)
    assert beat is not None
    assert len(beat) == 100


def test_extract_lead_features_keys():
    signal = np.sin(np.linspace(0, 20 * np.pi, 1200))
    rpeaks = np.arange(60, 1140, 100)
    beat = median_beat(signal, rpeaks, pre_samples=30, post_samples=70)
    feats = extract_lead_features(beat, pre_samples=30)
    expected = {'J_amp','ST40','ST80','ST_slope','R_amp','S_amp','QRS_dur','T_amp','J_to_R_ratio'}
    assert set(feats.keys()) == expected

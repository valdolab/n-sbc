"""Z-matrix data structure and feature-level match computation."""

from dataclasses import dataclass

import numpy as np


def compute_feature_match_ratios(test_bits, train_bits, feature_bit_widths):
    """Compute per-feature match ratios between a test sample and training samples.

    Parameters
    ----------
    test_bits : ndarray of shape (total_bit_width,)
    train_bits : ndarray of shape (n_samples, total_bit_width)
    feature_bit_widths : ndarray of shape (n_features,)

    Returns
    -------
    match_ratios : ndarray of shape (n_samples, n_features)
        Match ratio per feature, in [0, 1].
    """
    ends = np.cumsum(feature_bit_widths)
    starts = np.concatenate([[0], ends[:-1]])
    n_features = len(feature_bit_widths)
    n_samples = train_bits.shape[0]
    match_ratios = np.zeros((n_samples, n_features))

    for j in range(n_features):
        s, e = starts[j], ends[j]
        matches = test_bits[s:e] == train_bits[:, s:e]
        match_ratios[:, j] = np.mean(matches, axis=1)

    return match_ratios


@dataclass
class ZMatrix:
    """Explainability output from predict_explain().

    Attributes
    ----------
    z : ndarray of shape (n_test, n_train)
        Full Hamming similarity matrix.
    class_scores : ndarray of shape (n_test, n_classes)
        Sum of top-u similarities per class.
    predictions : ndarray of shape (n_test,)
        Predicted class labels.
    top_u_indices : list of dict
        Per test sample, maps class label to array of top-u training indices.
    feature_importances : ndarray of shape (n_test, n_features)
        Per-sample feature importance (mean match ratio across top-u neighbors).
    feature_bit_widths : ndarray of shape (n_features,)
        Number of bits per feature.
    classes : ndarray of shape (n_classes,)
        Class labels.
    n_value : int
        The u parameter used.
    x_test_encoded : ndarray of shape (n_test, total_bit_width)
        Binary-encoded test samples.
    x_train_encoded : ndarray of shape (n_train, total_bit_width)
        Binary-encoded training samples (reference, not copy).
    """

    z: np.ndarray
    class_scores: np.ndarray
    predictions: np.ndarray
    top_u_indices: list
    feature_importances: np.ndarray
    feature_bit_widths: np.ndarray
    classes: np.ndarray
    n_value: int
    x_test_encoded: np.ndarray
    x_train_encoded: np.ndarray

    @property
    def global_feature_importances(self):
        """Aggregate feature importances across all test samples."""
        return np.mean(self.feature_importances, axis=0)

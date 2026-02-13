"""Validation tests against MATLAB n-SBC results using the cryotherapy dataset."""

import os

import numpy as np
import pytest

from nsbc import NSBCClassifier


@pytest.fixture
def cryotherapy_data():
    """Load cryotherapy dataset."""
    csv_path = os.path.join(
        os.path.dirname(__file__), "..", "original_matlab_code", "cryotherapy.csv"
    )
    dataset = np.loadtxt(csv_path, delimiter=",")
    x = dataset[:, :-1]
    y = dataset[:, -1].astype(int)
    return x, y


def test_single_prediction(cryotherapy_data):
    """Test single prediction matches MATLAB (u=5, predict last sample -> 1)."""
    x, y = cryotherapy_data

    # Train on samples 0-87, predict sample 88 (last one, 0-indexed)
    x_train = x[:-1]
    y_train = y[:-1]
    x_test = x[-1:]

    clf = NSBCClassifier(n_value=5, decimals=2)
    clf.fit(x_train, y_train)
    prediction = clf.predict(x_test)

    assert prediction[0] == 1, f"Expected prediction 1, got {prediction[0]}"


def test_loocv_balanced_accuracy(cryotherapy_data):
    """Test LOOCV matches MATLAB (u=3, balanced accuracy ~92.17%)."""
    x, y = cryotherapy_data
    n_samples = x.shape[0]

    predictions = np.zeros(n_samples, dtype=int)
    for i in range(n_samples):
        # Leave one out
        train_mask = np.ones(n_samples, dtype=bool)
        train_mask[i] = False
        x_train = x[train_mask]
        y_train = y[train_mask]
        x_test = x[i : i + 1]

        clf = NSBCClassifier(n_value=3, decimals=2)
        clf.fit(x_train, y_train)
        predictions[i] = clf.predict(x_test)[0]

    # Compute balanced accuracy (per-class recall, then average)
    unique_labels = np.unique(y)
    per_class_recall = []
    for label in unique_labels:
        mask = y == label
        correct = np.sum(predictions[mask] == label)
        total = np.sum(mask)
        per_class_recall.append(correct / total if total > 0 else 0.0)

    balanced_acc = np.mean(per_class_recall) * 100

    # MATLAB target: 92.17% balanced accuracy
    assert (
        abs(balanced_acc - 92.17) < 1.0
    ), f"Balanced accuracy {balanced_acc:.2f}% deviates from MATLAB target 92.17%"

    # Also check per-class: sensitivity ~92.68%, specificity ~91.67%
    assert (
        abs(per_class_recall[0] * 100 - 91.67) < 1.5
    ), f"Specificity {per_class_recall[0]*100:.2f}% deviates from 91.67%"
    assert (
        abs(per_class_recall[1] * 100 - 92.68) < 1.5
    ), f"Sensitivity {per_class_recall[1]*100:.2f}% deviates from 92.68%"

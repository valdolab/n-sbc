"""Tests for predict_explain and ZMatrix."""

import numpy as np
import pytest

from nsbc import NSBCClassifier


@pytest.fixture
def fitted_model():
    data = np.loadtxt("tests/cryotherapy.csv", delimiter=",")
    x, y = data[:, :-1], data[:, -1]
    clf = NSBCClassifier(n_value=3, decimals=2)
    clf.fit(x[:80], y[:80])
    return clf, x, y


def test_predict_explain_matches_predict(fitted_model):
    clf, x, y = fitted_model
    x_test = x[80:]
    result = clf.predict_explain(x_test)
    pred_normal = clf.predict(x_test)
    np.testing.assert_array_equal(result.predictions, pred_normal)


def test_z_matrix_shapes(fitted_model):
    clf, x, y = fitted_model
    x_test = x[80:]
    result = clf.predict_explain(x_test)
    n_test = x_test.shape[0]
    n_train = 80
    n_features = x.shape[1]
    n_classes = len(np.unique(y[:80]))

    assert result.z.shape == (n_test, n_train)
    assert result.class_scores.shape == (n_test, n_classes)
    assert result.predictions.shape == (n_test,)
    assert result.feature_importances.shape == (n_test, n_features)
    assert len(result.top_u_indices) == n_test


def test_feature_importances_range(fitted_model):
    clf, x, y = fitted_model
    result = clf.predict_explain(x[80:])
    assert np.all(result.feature_importances >= 0)
    assert np.all(result.feature_importances <= 1)


def test_global_feature_importances(fitted_model):
    clf, x, y = fitted_model
    result = clf.predict_explain(x[80:])
    global_imp = result.global_feature_importances
    assert global_imp.shape == (x.shape[1],)
    expected = np.mean(result.feature_importances, axis=0)
    np.testing.assert_array_almost_equal(global_imp, expected)


def test_top_u_indices_size(fitted_model):
    clf, x, y = fitted_model
    result = clf.predict_explain(x[80:])
    for sample_dict in result.top_u_indices:
        for _label, indices in sample_dict.items():
            assert len(indices) <= clf.n_value


def test_get_pattern_importances(fitted_model):
    clf, x, y = fitted_model
    assert clf._engine.get_pattern_importances() is None
    clf.predict_explain(x[80:])
    imp = clf._engine.get_pattern_importances()
    assert imp is not None
    assert imp.shape == (x.shape[1],)


def test_compute_feature_match_ratios():
    from nsbc.tools.matrix_z import compute_feature_match_ratios

    test_bits = np.array([1, 0, 1, 1, 0], dtype=np.uint8)
    train_bits = np.array([[1, 0, 1, 1, 0], [0, 1, 0, 1, 0]], dtype=np.uint8)
    widths = np.array([2, 3], dtype=np.int32)
    ratios = compute_feature_match_ratios(test_bits, train_bits, widths)
    assert ratios.shape == (2, 2)
    np.testing.assert_array_almost_equal(ratios[0], [1.0, 1.0])
    np.testing.assert_array_almost_equal(ratios[1], [0.0, 2 / 3])

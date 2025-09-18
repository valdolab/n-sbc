"""Quick test to verify mock implementation works."""

import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from nsbc import NSBCClassifier


def test_basic_fit_predict():
    """Test basic fit and predict."""
    x, y = make_classification(
        n_samples=100, n_features=20, n_classes=2, random_state=42
    )
    clf = NSBCClassifier(random_state=42)
    clf.fit(x, y)
    predictions = clf.predict(x[:10])
    assert len(predictions) == 10
    print("✓ Basic fit/predict works")


def test_multiclass():
    """Test multiclass classification."""
    x, y = make_classification(
        n_samples=100, n_features=20, n_classes=3, n_informative=10, random_state=42
    )
    clf = NSBCClassifier(random_state=42)
    clf.fit(x, y)
    # predictions = clf.predict(x[:10])
    proba = clf.predict_proba(x[:10])
    assert proba.shape == (10, 3)
    assert np.allclose(proba.sum(axis=1), 1.0)
    print("✓ Multiclass classification works")


def test_pipeline():
    """Test in pipeline."""
    x, y = make_classification(n_samples=100, n_features=20, random_state=42)
    pipeline = Pipeline(
        [("scaler", StandardScaler()), ("classifier", NSBCClassifier(random_state=42))]
    )
    pipeline.fit(x, y)
    predictions = pipeline.predict(x[:10])
    assert len(predictions) == 10
    print("✓ Pipeline integration works")


def test_cross_validation():
    """Test cross-validation."""
    x, y = make_classification(n_samples=100, n_features=20, random_state=42)
    clf = NSBCClassifier(random_state=42)
    scores = cross_val_score(clf, x, y, cv=3)
    assert len(scores) == 3
    assert all(0 <= score <= 1 for score in scores)
    print(f"✓ Cross-validation works (scores: {scores})")


if __name__ == "__main__":
    test_basic_fit_predict()
    test_multiclass()
    test_pipeline()
    test_cross_validation()
    print("\n✅ All tests passed!")

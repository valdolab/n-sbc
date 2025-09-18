"""Test scikit-learn compatibility."""

from sklearn.datasets import make_classification, make_regression
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.utils.estimator_checks import check_estimator

from nsbc import NSBCClassifier


def test_sklearn_estimator_checks():
    """Test that our estimators pass scikit-learn's estimator checks."""
    # These checks ensure full compatibility with sklearn
    check_estimator(NSBCClassifier())


def test_classifier_with_cross_validation():
    """Test classifier with cross-validation."""
    x, y = make_classification(n_samples=100, n_features=20, random_state=42)
    clf = NSBCClassifier(random_state=42)

    scores = cross_val_score(clf, x, y, cv=3)
    assert len(scores) == 3
    assert all(0 <= score <= 1 for score in scores)


def test_classifier_with_pipeline():
    """Test classifier in a pipeline."""
    x, y = make_classification(n_samples=100, n_features=20, random_state=42)

    pipeline = Pipeline(
        [("scaler", StandardScaler()), ("classifier", NSBCClassifier(random_state=42))]
    )

    pipeline.fit(x, y)
    predictions = pipeline.predict(x)
    assert len(predictions) == len(y)


def test_classifier_with_grid_search():
    """Test classifier with GridSearchCV."""
    x, y = make_classification(n_samples=100, n_features=20, random_state=42)

    param_grid = {"n_components": [5, 10], "learning_rate": [0.01, 0.1]}

    clf = NSBCClassifier(random_state=42)
    grid_search = GridSearchCV(clf, param_grid, cv=3)
    grid_search.fit(x, y)

    assert hasattr(grid_search, "best_params_")
    assert hasattr(grid_search, "best_score_")


def test_regressor_with_cross_validation():
    """Test regressor with cross-validation."""
    x, y = make_regression(n_samples=100, n_features=20, random_state=42)

    scores = cross_val_score(x, y, cv=3, scoring="r2")
    assert len(scores) == 3

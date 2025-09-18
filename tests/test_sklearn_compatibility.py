"""Test scikit-learn compatibility."""

import pytest
import numpy as np
from sklearn.utils.estimator_checks import check_estimator
from sklearn.model_selection import cross_val_score, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import make_classification, make_regression

from nsbc import NSBCClassifier


def test_sklearn_estimator_checks():
    """Test that our estimators pass scikit-learn's estimator checks."""
    # These checks ensure full compatibility with sklearn
    check_estimator(NSBCClassifier())


def test_classifier_with_cross_validation():
    """Test classifier with cross-validation."""
    X, y = make_classification(n_samples=100, n_features=20, random_state=42)
    clf = NSBCClassifier(random_state=42)
    
    scores = cross_val_score(clf, X, y, cv=3)
    assert len(scores) == 3
    assert all(0 <= score <= 1 for score in scores)


def test_classifier_with_pipeline():
    """Test classifier in a pipeline."""
    X, y = make_classification(n_samples=100, n_features=20, random_state=42)
    
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('classifier', NSBCClassifier(random_state=42))
    ])
    
    pipeline.fit(X, y)
    predictions = pipeline.predict(X)
    assert len(predictions) == len(y)


def test_classifier_with_grid_search():
    """Test classifier with GridSearchCV."""
    X, y = make_classification(n_samples=100, n_features=20, random_state=42)
    
    param_grid = {
        'n_components': [5, 10],
        'learning_rate': [0.01, 0.1]
    }
    
    clf = NSBCClassifier(random_state=42)
    grid_search = GridSearchCV(clf, param_grid, cv=3)
    grid_search.fit(X, y)
    
    assert hasattr(grid_search, 'best_params_')
    assert hasattr(grid_search, 'best_score_')


def test_regressor_with_cross_validation():
    """Test regressor with cross-validation."""
    X, y = make_regression(n_samples=100, n_features=20, random_state=42)
    
    scores = cross_val_score(X, y, cv=3, scoring='r2')
    assert len(scores) == 3

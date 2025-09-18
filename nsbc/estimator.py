"""Main n-SBC estimator classes."""

from sklearn.base import ClassifierMixin
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.multiclass import check_classification_targets
from sklearn.utils.validation import check_array, check_is_fitted

from .base import BaseNSBC
from .engine import NSBCEngine


class NSBCClassifier(BaseNSBC, ClassifierMixin):
    """n-SBC Classifier.

    A novel machine learning classifier compatible with scikit-learn.

    Parameters
    ----------
    n_value : int, default=1
        Number value parameter for the model.
    decimals : int, default=2
        Number of decimal places for rounding.
    random_state : int or None, default=None
        Random state for reproducibility.
    verbose : int, default=0
        Verbosity level.

    Attributes
    ----------
    classes_ : ndarray of shape (n_classes,)
        The class labels.
    n_classes_ : int
        Number of classes.
    pattern_importances_ : ndarray of shape (n_features,)
        Feature importance scores.

    Examples
    --------
    >>> from nsbc import NSBCClassifier
    >>> from sklearn.datasets import make_classification
    >>> x, y = make_classification(n_samples=100, n_features=20, random_state=42)
    >>> clf = NSBCClassifier(random_state=42)
    >>> clf.fit(x, y)
    NSBCClassifier(random_state=42)
    >>> clf.predict(x[:5])
    array([0, 0, 1, 0, 1])
    """

    def _fit_internal(self, x, y):
        """Internal method to fit the classifier."""
        check_classification_targets(y)
        self._label_encoder = LabelEncoder()
        y_encoded = self._label_encoder.fit_transform(y)
        self.classes_ = self._label_encoder.classes_
        self.n_classes_ = len(self.classes_)
        self._engine = NSBCEngine(
            n_value=self.n_value,
            decimals=self.decimals,
            random_state=self.random_state,
        )

        self._engine.fit(x, y_encoded)
        self.pattern_importances_ = self._engine.get_pattern_importances()

    def _predict_internal(self, x):
        """Internal method to make predictions."""
        return self._engine.predict(x)

    def predict(self, x):
        """Predict class labels for samples in x.

        Parameters
        ----------
        x : array-like of shape (n_samples, n_features)
            The input samples.

        Returns
        -------
        y : ndarray of shape (n_samples,)
            The predicted class labels.
        """
        check_is_fitted(self, ["classes_", "_engine"])
        x = check_array(x, accept_sparse=False)
        if x.shape[1] != self.n_features_in_:
            raise ValueError(
                f"x has {x.shape[1]} features, "
                f"but NSBCClassifier is expecting {self.n_features_in_} features"
            )

        y_pred = self._predict_internal(x)
        return self._label_encoder.inverse_transform(y_pred)

    def predict_proba(self, x):
        """Predict class probabilities for samples in x.

        Parameters
        ----------
        x : array-like of shape (n_samples, n_features)
            The input samples.

        Returns
        -------
        proba : ndarray of shape (n_samples, n_classes)
            The class probabilities.
        """
        check_is_fitted(self, ["classes_", "_engine"])
        x = check_array(x, accept_sparse=False)
        if x.shape[1] != self.n_features_in_:
            raise ValueError(
                f"x has {x.shape[1]} features, "
                f"but NSBCClassifier is expecting {self.n_features_in_} features"
            )

        return self._engine.predict_proba(x)

    def score(self, x, y, sample_weight=None):
        """Return the mean accuracy on the given test data and labels.

        Parameters
        ----------
        x : array-like of shape (n_samples, n_features)
            Test samples.
        y : array-like of shape (n_samples,)
            True labels for x.
        sample_weight : array-like of shape (n_samples,), optional
            Sample weights.

        Returns
        -------
        score : float
            Mean accuracy.
        """
        from sklearn.metrics import accuracy_score

        return accuracy_score(y, self.predict(x), sample_weight=sample_weight)

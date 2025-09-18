"""Base classes for n-SBC estimators."""

from abc import ABC, abstractmethod

from sklearn.base import BaseEstimator
from sklearn.utils.validation import _check_sample_weight, check_x_y


class BaseNSBC(BaseEstimator, ABC):
    """Base class for n-SBC estimators.

    This class should not be used directly. Use derived classes instead.
    """

    def __init__(
        self,
        n_value=1,
        decimals=2,
        random_state=None,
        verbose=0,
    ):
        self.n_value = n_value
        self.decimals = decimals
        self.random_state = random_state
        self.verbose = verbose

    def _validate_params(self):
        """Validate input parameters."""
        if self.n_value <= 0:
            raise ValueError(f"n_value must be > 0, got {self.n_value}")
        if self.decimals < 0:
            raise ValueError(f"decimals must be >= 0, got {self.decimals}")

    @abstractmethod
    def _fit_internal(self, x, y):
        """Internal fitting method to be implemented by subclasses."""
        pass

    @abstractmethod
    def _predict_internal(self, x):
        """Internal prediction method to be implemented by subclasses."""
        pass

    def fit(self, x, y, sample_weight=None):
        """Fit the n-SBC model.

        Parameters
        ----------
        x : array-like of shape (n_samples, n_features)
            Training data.
        y : array-like of shape (n_samples,)
            Target values.
        sample_weight : array-like of shape (n_samples,), optional
            Sample weights.

        Returns
        -------
        self : object
            Returns self.
        """
        self._validate_params()
        x, y = check_x_y(x, y, accept_sparse=False)

        if sample_weight is not None:
            sample_weight = _check_sample_weight(sample_weight, x)

        self.n_features_in_ = x.shape[1]
        self.n_samples_ = x.shape[0]
        self._fit_internal(x, y)
        self.is_fitted_ = True
        return self

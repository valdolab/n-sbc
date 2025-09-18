"""Base classes for n-SBC estimators."""

import numpy as np
from abc import ABC, abstractmethod
from sklearn.base import BaseEstimator
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted


class BaseNSBC(BaseEstimator, ABC):
    """Base class for n-SBC estimators.
    
    This class should not be used directly. Use derived classes instead.
    """
    
    def __init__(
        self,
        n_value=10,
        random_state=None,
        verbose=0,
    ):
        self.n_value = n_value
        self.random_state = random_state
        self.verbose = verbose
    
    def _validate_params(self):
        """Validate input parameters."""
        if self.n_value <= 0:
            raise ValueError(f"n_value must be > 0, got {self.n_value}")
    
    @abstractmethod
    def _fit_internal(self, X, y):
        """Internal fitting method to be implemented by subclasses."""
        pass
    
    @abstractmethod
    def _predict_internal(self, X):
        """Internal prediction method to be implemented by subclasses."""
        pass
    
    def fit(self, X, y, sample_weight=None):
        """Fit the n-SBC model.
        
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
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
        X, y = check_X_y(X, y, accept_sparse=False)
        
        self.n_features_in_ = X.shape[1]
        self.n_samples_ = X.shape[0]
        self._fit_internal(X, y)
        self.is_fitted_ = True
        
        return self

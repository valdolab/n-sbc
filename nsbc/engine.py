"""Core engine for n-SBC algorithm."""

import pickle
from pathlib import Path

import numpy as np
from sklearn.utils import check_random_state

from nsbc.utils.encoders import MLBinaryEncoderVectorized


class NSBCEngine:
    """Core computational engine for n-SBC.

    This class implements the actual n-SBC algorithm.
    """

    def __init__(
        self,
        n_value=3,
        decimals=2,
        encoder_type="gray",
        random_state=None,
        verbose=False,
    ):
        """
        Initialize n-SBC Engine.

        Parameters
        ----------
        n_value : int, default=1
            Number of nearest neighbors to consider
        decimals : int, default=2
            Number of decimal places to preserve in encoding
        encoder_type : str, default='gray'
            Type of encoding ('gray' or 'binary')
        random_state : int, RandomState instance or None, default=None
            Controls randomness for reproducibility
        verbose : bool, default=False
            Whether to print progress information
        """
        self.n_value = n_value
        self.decimals = decimals
        self.encoder_type = encoder_type
        self.random_state = check_random_state(random_state)
        self.verbose = verbose

        self.encoder_ = None
        self.X_train_encoded_ = None
        self.y_train_ = None
        self.classes_ = None
        self.n_classes_ = None
        self._is_fitted = False

    def fit(self, x, y):
        """
        Fit the n-SBC model.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data
        y : array-like of shape (n_samples,)
            Target values

        Returns
        -------
        self : object
            Returns self for method chaining
        """
        x = np.asarray(x)
        y = np.asarray(y)

        if x.shape[0] != y.shape[0]:
            raise ValueError(
                f"X and y must have same number of samples. "
                f"Got X.shape[0]={x.shape[0]}, y.shape[0]={y.shape[0]}"
            )

        self.classes_ = np.unique(y)
        self.n_classes_ = len(self.classes_)

        if self.verbose:
            print(f"Fitting n-SBC with {x.shape[0]} samples, {x.shape[1]} features")
            print(f"Number of classes: {self.n_classes_}")

        # Step 1: Apply scale and encoder
        self.encoder_ = MLBinaryEncoderVectorized(
            encoder_type=self.encoder_type, verbose=self.verbose
        )

        # Step 2: Create X_train_encoded_ (M) matrix
        self.X_train_encoded_ = self.encoder_.fit_transform(
            x, num_decimals=self.decimals
        )
        if isinstance(self.X_train_encoded_, tuple):
            self.X_train_encoded_ = self.X_train_encoded_[0]

        self.y_train_ = y.copy()
        self._is_fitted = True

        if self.verbose:
            print(f"Training complete. Encoded shape: {self.X_train_encoded_.shape}")
        return self

    def _transform(self, x):
        """Transform input to hidden representation."""
        # Mock: fake _transform data
        n = x.shape[0]
        x_aug = np.hstack([np.ones((n, 1)), x])
        return np.dot(x_aug, self.coef_)

    def _decision_function(self, hidden):
        """Compute decision function from hidden representation."""
        # Mock: fake decision data
        return hidden

    def predict(self, x):
        """Make predictions."""
        # TODO: Implement actual n-SBC prediction algorithm
        #

        # Mock: fake predict data
        hidden = self._transform(x)
        decision = self._decision_function(hidden)

        if self.n_classes_ == 2:
            # Binary classification
            predictions = (decision.flatten() > 0.5).astype(int)
            return self.classes_[predictions]
        else:
            # Multi-class: argmax
            return self.classes_[np.argmax(decision, axis=1)]

    def predict_proba(self, x):
        """Predict probabilities (for classification only)."""
        # TODO: Implement actual n-SBC probability estimation
        #

        # Mock: fake predict data
        hidden = self._transform(x)
        decision = self._decision_function(hidden)

        if self.n_classes_ == 2:
            # Binary classification
            decision = decision.flatten()
            decision = np.clip(decision, -500, 500)
            proba_pos = 1 / (1 + np.exp(-decision))
            proba_neg = 1 - proba_pos
            return np.column_stack([proba_neg, proba_pos])
        else:
            # Multi-class - softmax
            decision_shifted = decision - np.max(decision, axis=1, keepdims=True)
            exp_decision = np.exp(decision_shifted)
            return exp_decision / np.sum(exp_decision, axis=1, keepdims=True)

    def get_pattern_importances(self):
        """Calculate feature importances."""
        # TODO: Implement actual pattern importance calculation
        #

        # Mock: return absolute mean of coefficients (excluding intercept)
        if self.coef_ is None:
            return None

        if self.n_classes_ == 2:
            # Binary: single set of coefficients
            return np.abs(self.coef_[1:])
        else:
            # Multi-class: average across all classes
            return np.mean(np.abs(self.coef_[1:]), axis=1)

    def save(self, filepath: str = "./model.pkl"):
        """
        Save the fitted model to a pickle file.

        Parameters
        ----------
        filepath : str
            Path where to save the model
        """
        if not self._is_fitted:
            raise ValueError("Cannot save unfitted model. Call fit() first.")
        filepath = Path(filepath)

        filepath.parent.mkdir(parents=True, exist_ok=True)
        model_state = {
            "n_value": self.n_value,
            "decimals": self.decimals,
            "encoder_type": self.encoder_type,
            "encoder_params": self.encoder_.params.to_dict()
            if self.encoder_.params
            else None,
            "X_train_encoded": self.X_train_encoded_,
            "y_train": self.y_train_,
            "classes": self.classes_,
            "n_classes": self.n_classes_,
            "random_state": self.random_state,
        }

        with open(filepath, "wb") as f:
            pickle.dump(model_state, f)

        if self.verbose:
            print(f"Model saved to {filepath}")

    @classmethod
    def load(cls, filepath, verbose=False):
        """
        Load a fitted model from a pickle file.

        Parameters
        ----------
        filepath : str or Path
            Path to the saved model
        verbose : bool, default=False
            Whether to print loading information

        Returns
        -------
        model : NSBCEngine
            Loaded model ready for prediction
        """
        from nsbc.utils.encoders import EncodingParams, MLBinaryEncoderVectorized

        filepath = Path(filepath)
        if not filepath.exists():
            raise FileNotFoundError(f"Model file not found: {filepath}")

        with open(filepath, "rb") as f:
            model_state = pickle.load(f)

        model = cls(
            n_value=model_state["n_value"],
            decimals=model_state["decimals"],
            encoder_type=model_state["encoder_type"],
            random_state=model_state["random_state"],
            verbose=verbose,
        )

        model.encoder_ = MLBinaryEncoderVectorized(
            encoder_type=model_state["encoder_type"], verbose=verbose
        )

        if model_state["encoder_params"]:
            model.encoder_.params = EncodingParams.from_dict(
                model_state["encoder_params"]
            )
            model.encoder_.is_fitted = True

        model.X_train_encoded_ = model_state["X_train_encoded"]
        model.y_train_ = model_state["y_train"]
        model.classes_ = model_state["classes"]
        model.n_classes_ = model_state["n_classes"]
        model._is_fitted = True
        if verbose:
            print(f"Model loaded from {filepath}")
            print(f"Training set shape: {model.X_train_encoded_.shape}")
            print(f"Number of classes: {model.n_classes_}")
        return model

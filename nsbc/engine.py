"""Core engine for n-SBC algorithm."""

import pickle
from pathlib import Path

import numpy as np
from sklearn.utils import check_random_state

from nsbc.tools.matrix_z import ZMatrix, compute_feature_match_ratios
from nsbc.utils.encoders import MLBinaryEncoderVectorized


class NSBCEngine:
    """Core computational engine for n-SBC.

    This class implements the actual n-SBC algorithm.
    """

    def __init__(
        self,
        n_value=3,
        decimals=2,
        factor=10,
        encoder_type="gray",
        random_state=None,
        verbose=False,
    ):
        """
        Initialize n-SBC Engine.

        Parameters
        ----------
        n_value : int, default=3
            Number of top neighbors to sum per class
        decimals : int, default=2
            Number of decimal places to preserve in encoding
        factor : int, default=10
            Multiplicative factor to convert rounded data to integers
        encoder_type : str, default='gray'
            Type of encoding ('gray' or 'binary')
        random_state : int, RandomState instance or None, default=None
            Controls randomness for reproducibility
        verbose : bool, default=False
            Whether to print progress information
        """
        self.n_value = n_value
        self.decimals = decimals
        self.factor = factor
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

        Replicates nsbc_train.m: preprocesses data to Gray-coded binary
        and stores it for lazy classification.

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

        # Apply scale and encoder
        self.encoder_ = MLBinaryEncoderVectorized(
            encoder_type=self.encoder_type,
            verbose=self.verbose,
        )

        # Create X_train_encoded_ (M) matrix
        self.X_train_encoded_ = self.encoder_.fit_transform(
            x, num_decimals=self.decimals, factor=self.factor
        )
        if isinstance(self.X_train_encoded_, tuple):
            self.X_train_encoded_ = self.X_train_encoded_[0]

        self.y_train_ = y.copy()
        self._is_fitted = True

        if self.verbose:
            print(f"Training complete. Encoded shape: {self.X_train_encoded_.shape}")
        return self

    def predict(self, x):
        """
        Make predictions using the n-SBC algorithm.

        Replicates nsbc_predict.m.

        Parameters
        ----------
        x : array-like of shape (n_samples, n_features)
            Samples to predict.

        Returns
        -------
        predictions : ndarray of shape (n_samples,)
            Predicted class labels.
        """
        x = np.asarray(x)
        x_test_encoded = self.encoder_.transform(x)
        if isinstance(x_test_encoded, tuple):
            x_test_encoded = x_test_encoded[0]

        n_test = x_test_encoded.shape[0]
        predictions = np.zeros(n_test, dtype=int)
        for i in range(n_test):
            predictions[i] = self._classify_sample(x_test_encoded[i])
        return predictions

    def _classify_sample(self, test_pattern):
        """
        Classify a single sample using Hamming similarity scoring.

        Replicates nsbc_classify_sample.m: computes Hamming similarity
        to all training patterns, sums the top-u per class, and assigns
        the class with the highest sum.

        Parameters
        ----------
        test_pattern : ndarray of shape (n_bits,)
            Binary-encoded test sample.

        Returns
        -------
        predicted_label : int
            Predicted class label.
        """
        n_bits = test_pattern.shape[0]
        # Hamming distances vectorized: count differing bits per training sample
        distances = np.count_nonzero(self.X_train_encoded_ != test_pattern, axis=1)
        similarities = n_bits - distances

        class_scores = np.zeros(self.n_classes_)
        for k, label in enumerate(self.classes_):
            class_sims = similarities[self.y_train_ == label]
            sorted_desc = np.sort(class_sims)[::-1]
            # Sum top-u similarities (or all if fewer than u)
            top_u = min(self.n_value, len(sorted_desc))
            class_scores[k] = np.sum(sorted_desc[:top_u])

        return self.classes_[np.argmax(class_scores)]

    def predict_proba(self, x):
        """
        Predict class probabilities by normalizing similarity scores.

        Parameters
        ----------
        x : array-like of shape (n_samples, n_features)
            Samples to predict.

        Returns
        -------
        proba : ndarray of shape (n_samples, n_classes)
            Class probability estimates.
        """
        x = np.asarray(x)
        x_test_encoded = self.encoder_.transform(x)
        if isinstance(x_test_encoded, tuple):
            x_test_encoded = x_test_encoded[0]

        n_test = x_test_encoded.shape[0]
        n_bits = x_test_encoded.shape[1]
        proba = np.zeros((n_test, self.n_classes_))

        for i in range(n_test):
            distances = np.count_nonzero(
                self.X_train_encoded_ != x_test_encoded[i], axis=1
            )
            similarities = n_bits - distances

            class_scores = np.zeros(self.n_classes_)
            for k, label in enumerate(self.classes_):
                class_sims = similarities[self.y_train_ == label]
                sorted_desc = np.sort(class_sims)[::-1]
                top_u = min(self.n_value, len(sorted_desc))
                class_scores[k] = np.sum(sorted_desc[:top_u])

            total = np.sum(class_scores)
            if total > 0:
                proba[i] = class_scores / total
            else:
                proba[i] = 1.0 / self.n_classes_

        return proba

    def predict_explain(self, x):
        """Predict with full explainability output.

        Parameters
        ----------
        x : array-like of shape (n_samples, n_features)

        Returns
        -------
        ZMatrix
            Dataclass with z-matrix, class scores, predictions,
            top-u indices, and feature importances.
        """
        x = np.asarray(x)
        x_test_encoded = self.encoder_.transform(x)
        if isinstance(x_test_encoded, tuple):
            x_test_encoded = x_test_encoded[0]

        n_test = x_test_encoded.shape[0]
        n_train = self.X_train_encoded_.shape[0]
        n_bits = x_test_encoded.shape[1]
        n_features = self.encoder_.params.n_features
        feature_bit_widths = self.encoder_.params.feature_bit_widths

        z = np.zeros((n_test, n_train), dtype=np.int32)
        class_scores = np.zeros((n_test, self.n_classes_))
        predictions = np.zeros(n_test, dtype=int)
        top_u_indices = []
        feature_importances = np.zeros((n_test, n_features))

        for i in range(n_test):
            distances = np.count_nonzero(
                self.X_train_encoded_ != x_test_encoded[i], axis=1
            )
            z[i] = n_bits - distances

            sample_top_u = {}
            for k, label in enumerate(self.classes_):
                mask = self.y_train_ == label
                class_indices = np.where(mask)[0]
                class_sims = z[i, mask]
                sorted_order = np.argsort(-class_sims)
                top_u = min(self.n_value, len(sorted_order))
                top_indices = class_indices[sorted_order[:top_u]]
                sample_top_u[label] = top_indices
                class_scores[i, k] = np.sum(class_sims[sorted_order[:top_u]])

            pred_class = self.classes_[np.argmax(class_scores[i])]
            predictions[i] = pred_class

            top_train_idx = sample_top_u[pred_class]
            ratios = compute_feature_match_ratios(
                x_test_encoded[i],
                self.X_train_encoded_[top_train_idx],
                feature_bit_widths,
            )
            feature_importances[i] = np.mean(ratios, axis=0)
            top_u_indices.append(sample_top_u)

        result = ZMatrix(
            z=z,
            class_scores=class_scores,
            predictions=predictions,
            top_u_indices=top_u_indices,
            feature_importances=feature_importances,
            feature_bit_widths=feature_bit_widths,
            classes=self.classes_,
            n_value=self.n_value,
            x_test_encoded=x_test_encoded,
            x_train_encoded=self.X_train_encoded_,
        )
        self._last_z_matrix = result
        return result

    def get_pattern_importances(self):
        """Return global feature importances if predict_explain was called."""
        if hasattr(self, "_last_z_matrix") and self._last_z_matrix is not None:
            return self._last_z_matrix.global_feature_importances
        return None

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
            encoder_type=model_state["encoder_type"],
            verbose=verbose,
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

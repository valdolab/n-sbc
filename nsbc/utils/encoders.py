import json
import pickle
import warnings
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import numpy as np
from tqdm import tqdm


@dataclass
class EncodingParams:
    """Parameters learned during fitting that must be preserved for transform"""

    num_decimals: int = 2
    decimal_factor: int = 1
    feature_mins: np.ndarray = None
    feature_maxs: np.ndarray = None
    feature_ranges: np.ndarray = None
    feature_bit_widths: np.ndarray = None
    total_bit_width: int = 0
    n_features: int = 0
    offset_value: int = 0

    def to_dict(self) -> Dict:
        """Convert to dictionary for serialization"""
        params = {}
        for key, value in self.__dict__.items():
            if isinstance(value, np.ndarray):
                params[key] = value.tolist()
            elif isinstance(value, (np.integer, np.int64, np.int32)):
                params[key] = int(value)
            elif isinstance(value, (np.floating, np.float64, np.float32)):
                params[key] = float(value)
            else:
                params[key] = value
        return params

    @classmethod
    def from_dict(cls, params_dict: Dict) -> "EncodingParams":
        """Create from dictionary"""
        params = cls()
        for key, value in params_dict.items():
            if key in [
                "feature_mins",
                "feature_maxs",
                "feature_ranges",
                "feature_bit_widths",
            ]:
                setattr(params, key, np.array(value) if value is not None else None)
            else:
                setattr(params, key, value)
        return params

    def save(self, filepath: str):
        """Save parameters to file"""
        if filepath.endswith(".json"):
            with open(filepath, "w") as f:
                json.dump(self.to_dict(), f, indent=2)
        else:
            with open(filepath, "wb") as f:
                pickle.dump(self, f)

    @classmethod
    def load(cls, filepath: str) -> "EncodingParams":
        """Load parameters from file"""
        if filepath.endswith(".json"):
            with open(filepath) as f:
                return cls.from_dict(json.load(f))
        else:
            with open(filepath, "rb") as f:
                return pickle.load(f)


class BinaryEncoderBase(ABC):
    """Abstract base class for binary encoding methods"""

    @abstractmethod
    def encode_bits(self, binary_str: str) -> str:
        """Encode a binary string"""
        pass

    @abstractmethod
    def decode_bits(self, encoded_str: str) -> str:
        """Decode an encoded string back to binary"""
        pass

    @property
    @abstractmethod
    def name(self) -> str:
        """Return encoder name"""
        pass


class BinaryEncoder(BinaryEncoderBase):
    """Plain binary encoder (no transformation)"""

    def encode_bits(self, binary_str: str) -> str:
        return binary_str

    def decode_bits(self, encoded_str: str) -> str:
        return encoded_str

    @property
    def name(self) -> str:
        return "Binary"


class GrayCodeEncoder(BinaryEncoderBase):
    """Gray code encoder using XOR-based conversion"""

    def encode_bits(self, binary_str: str) -> str:
        """Convert binary to Gray code"""
        if not binary_str:
            return ""

        gray = [binary_str[0]]
        for i in range(1, len(binary_str)):
            bit = str(int(binary_str[i - 1]) ^ int(binary_str[i]))
            gray.append(bit)
        return "".join(gray)

    def decode_bits(self, gray_str: str) -> str:
        """Convert Gray code back to binary"""
        if not gray_str:
            return ""

        binary = [gray_str[0]]
        for i in range(1, len(gray_str)):
            bit = str(int(binary[-1]) ^ int(gray_str[i]))
            binary.append(bit)
        return "".join(binary)

    @property
    def name(self) -> str:
        return "GrayCode"


class MLBinaryEncoder:
    """
    Binary encoder for Machine Learning with fit/transform pattern.
    Learns normalization parameters during fit and applies them consistently.
    """

    def __init__(
        self, encoder_type: str = "gray", chunk_size: int = 10000, verbose: bool = True
    ):
        """
        Initialize the ML Binary Encoder.

        Args:
            encoder_type: Type of encoding ('gray' or 'binary')
            chunk_size: Size of chunks for processing large datasets
            verbose: Whether to print progress information
        """
        self.encoder_type = encoder_type.lower()
        self.chunk_size = chunk_size
        self.verbose = verbose

        if self.encoder_type == "gray":
            self.encoder = GrayCodeEncoder()
        elif self.encoder_type == "binary":
            self.encoder = BinaryEncoder()
        else:
            raise ValueError(
                f"Unknown encoder type: {encoder_type}. Use 'gray' or 'binary'"
            )

        # Parameters learned during fit
        self.params: Optional[EncodingParams] = None
        self.is_fitted = False

    def fit(self, x: np.ndarray, num_decimals: int = 2) -> "MLBinaryEncoder":
        """
        Fit the encoder to training data, learning normalization parameters.

        Args:
            X: Training data (n_samples x n_features)
            num_decimals: Number of decimal places to preserve

        Returns:
            self: Fitted encoder
        """
        if self.verbose:
            print(
                f"Fitting {self.encoder.name} encoder on {x.shape[0]} samples with {x.shape[1]} features"
            )

        n_samples, n_features = x.shape
        self.params = EncodingParams()
        self.params.num_decimals = num_decimals
        self.params.decimal_factor = 10**num_decimals
        self.params.n_features = n_features

        x_scaled = np.round(x * self.params.decimal_factor).astype(np.int64)
        self.params.feature_mins = np.min(x_scaled, axis=0)
        self.params.feature_maxs = np.max(x_scaled, axis=0)
        self.params.feature_ranges = self.params.feature_maxs - self.params.feature_mins

        # We add a small buffer (1) to ensure strictly positive values
        self.params.offset_value = (
            abs(np.min(self.params.feature_mins)) + 1
            if np.min(self.params.feature_mins) < 0
            else 0
        )
        x_positive = x_scaled + self.params.offset_value
        self.params.feature_bit_widths = np.zeros(n_features, dtype=np.int32)
        for i in range(n_features):
            max_val = np.max(x_positive[:, i])
            # Calculate bits needed: ceil(log2(max_val + 1))
            if max_val > 0:
                self.params.feature_bit_widths[i] = int(np.ceil(np.log2(max_val + 1)))
            else:
                self.params.feature_bit_widths[i] = 1

        self.params.total_bit_width = int(np.sum(self.params.feature_bit_widths))

        if self.verbose:
            print("Normalization learned:")
            print(f"  - Decimal factor: {self.params.decimal_factor}")
            print(f"  - Offset value: {self.params.offset_value}")
            print(
                f"  - Feature ranges: min={self.params.feature_mins.min()}, max={self.params.feature_maxs.max()}"
            )
            print(f"  - Bit widths per feature: {self.params.feature_bit_widths}")
            print(f"  - Total bit width: {self.params.total_bit_width}")

        self.is_fitted = True
        return self

    def transform(
        self, x: np.ndarray, normalized: bool = False, warning_flag: bool = False
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Transform data using learned parameters.

        Args:
            X: Data to transform (n_samples x n_features)

        Returns:
            Tuple of (encoded_binary_matrix, normalized_integer_data)
        """
        self._validate_for_transform(x)
        n_samples = x.shape[0]
        if self.verbose:
            print(f"Transforming {n_samples} samples")

        x_normalized = self._normalize_and_clip(x, warning_flag)
        encoded = self._encode_data(x_normalized, n_samples)
        if self.verbose:
            print(f"Output shape: {encoded.shape}")
        return (encoded, x_normalized) if normalized else encoded

    def _validate_for_transform(self, x: np.ndarray) -> None:
        """Validate input data for transformation."""
        if not self.is_fitted:
            raise ValueError(
                "Encoder must be fitted before transform. Call fit() first."
            )
        n_features = x.shape[1]
        if n_features != self.params.n_features:
            raise ValueError(
                f"Expected {self.params.n_features} features, got {n_features}"
            )

    def _normalize_and_clip(self, x: np.ndarray, warning_flag: bool) -> np.ndarray:
        """Normalize and clip features to training range."""
        x_scaled = np.round(x * self.params.decimal_factor).astype(np.int64)
        x_normalized = x_scaled + self.params.offset_value
        for i in range(x.shape[1]):
            x_normalized[:, i] = self._clip_feature(
                x_scaled[:, i], x_normalized[:, i], i, warning_flag
            )
        return x_normalized

    def _clip_feature(
        self,
        x_scaled_col: np.ndarray,
        x_normalized_col: np.ndarray,
        feature_idx: int,
        warning_flag: bool,
    ) -> np.ndarray:
        """Clip a single feature to its training range."""
        feature_min = np.min(x_scaled_col)
        feature_max = np.max(x_scaled_col)

        if self._is_out_of_range(feature_min, feature_max, feature_idx):
            if warning_flag:
                warnings.warn(
                    f"Feature {feature_idx}: values [{feature_min}, {feature_max}] outside training range "
                    f"[{self.params.feature_mins[feature_idx]}, {self.params.feature_maxs[feature_idx]}]. "
                    "Clipping to training range.",
                    category=UserWarning,
                    stacklevel=3,
                )
            return np.clip(
                x_normalized_col,
                self.params.feature_mins[feature_idx] + self.params.offset_value,
                self.params.feature_maxs[feature_idx] + self.params.offset_value,
            )
        return x_normalized_col

    def _is_out_of_range(
        self, feature_min: int, feature_max: int, feature_idx: int
    ) -> bool:
        """Check if feature values are outside training range."""
        return (
            feature_min < self.params.feature_mins[feature_idx]
            or feature_max > self.params.feature_maxs[feature_idx]
        )

    def _encode_data(self, x_normalized: np.ndarray, n_samples: int) -> np.ndarray:
        """Encode normalized data in chunks."""
        encoded = np.zeros((n_samples, self.params.total_bit_width), dtype=np.uint8)
        n_chunks = int(np.ceil(n_samples / self.chunk_size))

        pbar = (
            tqdm(range(n_chunks), desc="Encoding") if self.verbose else range(n_chunks)
        )

        for chunk_idx in pbar:
            start_idx = chunk_idx * self.chunk_size
            end_idx = min((chunk_idx + 1) * self.chunk_size, n_samples)
            encoded[start_idx:end_idx] = self._encode_chunk(
                x_normalized[start_idx:end_idx]
            )

        return encoded

    def fit_transform(
        self, x: np.ndarray, num_decimals: int = 2
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Fit encoder and transform data in one step.

        Args:
            X: Training data (n_samples x n_features)
            num_decimals: Number of decimal places to preserve

        Returns:
            Tuple of (encoded_binary_matrix, normalized_integer_data)
        """
        self.fit(x, num_decimals)
        return self.transform(x)

    def _encode_chunk(self, chunk: np.ndarray) -> np.ndarray:
        """
        Encode a chunk of normalized data.

        Args:
            chunk: Normalized positive integer data

        Returns:
            Binary encoded matrix
        """
        n_samples = chunk.shape[0]
        encoded = np.zeros((n_samples, self.params.total_bit_width), dtype=np.uint8)

        col_offset = 0
        for feature_idx in range(self.params.n_features):
            width = self.params.feature_bit_widths[feature_idx]

            for sample_idx in range(n_samples):
                # Convert to binary
                value = int(chunk[sample_idx, feature_idx])
                binary_str = format(value, f"0{width}b")
                encoded_str = self.encoder.encode_bits(binary_str)
                for bit_idx, bit in enumerate(encoded_str):
                    encoded[sample_idx, col_offset + bit_idx] = int(bit)

            col_offset += width

        return encoded

    def inverse_transform(self, x_encoded: np.ndarray) -> np.ndarray:
        """
        Inverse transform: decode binary back to original scale.

        Args:
            x_encoded: Binary encoded matrix

        Returns:
            Original data (with potential precision loss from discretization)
        """
        if not self.is_fitted:
            raise ValueError("Encoder must be fitted before inverse_transform.")

        n_samples = x_encoded.shape[0]
        x_encoded = np.zeros((n_samples, self.params.n_features))

        col_offset = 0
        for feature_idx in range(self.params.n_features):
            width = self.params.feature_bit_widths[feature_idx]

            for sample_idx in range(n_samples):
                encoded_bits = x_encoded[sample_idx, col_offset : col_offset + width]
                encoded_str = "".join(map(str, encoded_bits.astype(int)))
                binary_str = self.encoder.decode_bits(encoded_str)
                value = int(binary_str, 2) if binary_str else 0
                value = value - self.params.offset_value
                x_encoded[sample_idx, feature_idx] = value / self.params.decimal_factor

            col_offset += width

        return x_encoded

    def save_params(self, filepath: str):
        """
        Save encoding parameters to file.

        Args:
            filepath: Path to save parameters (use .json or .pkl extension)
        """
        if not self.is_fitted:
            raise ValueError("Encoder must be fitted before saving parameters.")

        self.params.save(filepath)
        if self.verbose:
            print(f"Encoding parameters saved to {filepath}")

    def load_params(self, filepath: str):
        """
        Load encoding parameters from file.

        Args:
            filepath: Path to load parameters from
        """
        self.params = EncodingParams.load(filepath)
        self.is_fitted = True
        if self.verbose:
            print(f"Encoding parameters loaded from {filepath}")

    def get_params(self) -> Dict[str, Any]:
        """Get encoder configuration parameters"""
        return {
            "encoder_type": self.encoder_type,
            "chunk_size": self.chunk_size,
            "verbose": self.verbose,
            "is_fitted": self.is_fitted,
            "encoding_params": self.params.to_dict() if self.params else None,
        }


class MLBinaryEncoderVectorized(MLBinaryEncoder):
    """
    Optimized vectorized version of MLBinaryEncoder for maximum performance.
    """

    def _encode_chunk(self, chunk: np.ndarray) -> np.ndarray:
        """
        Vectorized encoding of a chunk.
        """
        n_samples, n_features = chunk.shape
        encoded = np.zeros((n_samples, self.params.total_bit_width), dtype=np.uint8)

        col_offset = 0
        for feature_idx in tqdm(range(n_features)):
            width = self.params.feature_bit_widths[feature_idx]
            feature_values = chunk[:, feature_idx].astype(np.int64)
            binary_matrix = np.zeros((n_samples, width), dtype=np.uint8)
            for bit_pos in range(width):
                binary_matrix[:, width - 1 - bit_pos] = (feature_values >> bit_pos) & 1
            if self.encoder_type == "binary":
                encoded[:, col_offset : col_offset + width] = binary_matrix
            else:
                # Gray code encoding
                gray_matrix = np.zeros_like(binary_matrix)
                gray_matrix[:, 0] = binary_matrix[:, 0]
                for i in range(1, width):
                    gray_matrix[:, i] = binary_matrix[:, i - 1] ^ binary_matrix[:, i]
                encoded[:, col_offset : col_offset + width] = gray_matrix

            col_offset += width

        return encoded

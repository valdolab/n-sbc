import json
import pickle
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import numpy as np
from tqdm import tqdm


@dataclass
class EncodingParams:
    """Parameters learned during fitting that must be preserved for transform"""

    num_decimals: int = 2
    factor: int = 10
    feature_bit_widths: np.ndarray = None
    total_bit_width: int = 0
    n_features: int = 0
    min_val: float = 0.0

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
            if key in ["feature_bit_widths"]:
                setattr(params, key, np.array(value) if value is not None else None)
            elif hasattr(params, key):
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

    Normalization replicates nsbc_normalize.m:
      1. Shift data by abs(global_min) so all values >= 0
      2. Round to num_decimals, then multiply by 10 and round to integer
      3. Clamp negatives to 0
    """

    def __init__(
        self,
        encoder_type: str = "gray",
        chunk_size: int = 10000,
        verbose: bool = True,
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

    def fit(
        self, x: np.ndarray, num_decimals: int = 2, factor: int = 10
    ) -> "MLBinaryEncoder":
        """
        Fit the encoder to training data, learning normalization parameters.

        Replicates nsbc_normalize.m (training mode) + nsbc_togray.m (bit widths).

        Args:
            x: Training data (n_samples x n_features)
            num_decimals: Number of decimal places to preserve
            factor: Multiplicative factor to convert rounded data to integers (default 10)

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
        self.params.factor = factor
        self.params.n_features = n_features

        # Global minimum across all features (single scalar)
        # MATLAB: params.min_val = min(min(data))
        global_min = np.min(x)
        self.params.min_val = float(global_min)

        # Shift to positive, round, then convert to integers
        # MATLAB: data_positive = data + abs(params.min_val)
        #         data_positive = round(data_positive, num_decimals)
        #         data_int = round(data_positive * factorm)
        #         data_int(data_int < 0) = 0
        data_positive = x + abs(global_min)
        data_rounded = np.round(data_positive, num_decimals)
        data_int = np.floor(data_rounded * factor + 0.5).astype(np.int64)
        data_int[data_int < 0] = 0

        # Bit widths from max value per feature
        # MATLAB: feature_bin_lengths(i) = length(dec2bin(max_val))
        self.params.feature_bit_widths = np.zeros(n_features, dtype=np.int32)
        for i in range(n_features):
            max_val = int(np.max(data_int[:, i]))
            if max_val > 0:
                self.params.feature_bit_widths[i] = max_val.bit_length()
            else:
                self.params.feature_bit_widths[i] = 1

        self.params.total_bit_width = int(np.sum(self.params.feature_bit_widths))

        if self.verbose:
            print("Normalization learned:")
            print(f"  - Bit widths per feature: {self.params.feature_bit_widths}")
            print(f"  - Total bit width: {self.params.total_bit_width}")

        self.is_fitted = True
        return self

    def transform(
        self, x: np.ndarray, normalized: bool = False
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Transform data using learned parameters.

        Replicates nsbc_togray.m (transform mode with existing params).

        Args:
            x: Data to transform (n_samples x n_features)
            normalized: If True, also return the normalized integer data

        Returns:
            Encoded binary matrix, or tuple of (encoded, normalized) if normalized=True
        """
        self._validate_for_transform(x)
        n_samples = x.shape[0]
        if self.verbose:
            print(f"Transforming {n_samples} samples")

        x_normalized = self._normalize(x)
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

    def _normalize(self, x: np.ndarray) -> np.ndarray:
        """
        Normalize data using learned parameters.

        Replicates nsbc_normalize.m (with saved params) + nsbc_togray.m clamping.
        """
        # MATLAB: data_positive = data + abs(params.min_val)
        data_positive = x + abs(self.params.min_val)
        data_rounded = np.round(data_positive, self.params.num_decimals)
        # MATLAB: data_int = round(data_positive * factorm)
        x_normalized = np.floor(data_rounded * self.params.factor + 0.5).astype(
            np.int64
        )
        # MATLAB: data_int(data_int < 0) = 0
        x_normalized[x_normalized < 0] = 0
        # MATLAB: data_int(:, i) = min(data_int(:, i), 2^feature_bin_lengths(i) - 1)
        for i in range(x.shape[1]):
            max_allowed = (1 << int(self.params.feature_bit_widths[i])) - 1
            x_normalized[:, i] = np.minimum(x_normalized[:, i], max_allowed)
        return x_normalized

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
        self, x: np.ndarray, num_decimals: int = 2, factor: int = 10
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Fit encoder and transform data in one step.

        Args:
            x: Training data (n_samples x n_features)
            num_decimals: Number of decimal places to preserve
            factor: Multiplicative factor to convert rounded data to integers (default 10)

        Returns:
            Tuple of (encoded_binary_matrix, normalized_integer_data)
        """
        self.fit(x, num_decimals, factor)
        return self.transform(x)

    def _encode_chunk(self, chunk: np.ndarray) -> np.ndarray:
        """
        Encode a chunk of normalized data.

        Replicates nsbc_togray.m binary conversion + Gray encoding loop.

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
        result = np.zeros((n_samples, self.params.n_features))

        col_offset = 0
        for feature_idx in range(self.params.n_features):
            width = self.params.feature_bit_widths[feature_idx]

            for sample_idx in range(n_samples):
                encoded_bits = x_encoded[sample_idx, col_offset : col_offset + width]
                encoded_str = "".join(map(str, encoded_bits.astype(int)))
                binary_str = self.encoder.decode_bits(encoded_str)
                value = int(binary_str, 2) if binary_str else 0
                # Reverse: int_val = round((x + abs(min_val)) * factor)
                # so x = int_val / factor - abs(min_val)
                result[sample_idx, feature_idx] = value / self.params.factor - abs(
                    self.params.min_val
                )

            col_offset += width

        return result

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
        feature_iter = (
            tqdm(range(n_features), desc="Encoding features")
            if self.verbose
            else range(n_features)
        )
        for feature_idx in feature_iter:
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

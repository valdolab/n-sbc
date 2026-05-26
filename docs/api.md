# API Reference

## NSBCClassifier

```python
from nsbc import NSBCClassifier
```

The main estimator class, fully compatible with scikit-learn.

### Constructor

```python
NSBCClassifier(n_value=3, decimals=2, factor=10, random_state=None, verbose=0)
```

**Parameters:**

| Parameter      | Type         | Default | Description                                          |
|----------------|--------------|---------|------------------------------------------------------|
| `n_value`      | int          | 3       | Number of top-*u* similar training samples per class |
| `decimals`     | int          | 2       | Decimal places for rounding during normalization     |
| `factor`       | int          | 10      | Multiplicative factor applied after rounding         |
| `random_state` | int or None  | None    | Random state for reproducibility                     |
| `verbose`      | int          | 0       | Verbosity level                                      |

### Methods

#### `fit(X, y)`

Fit the classifier on training data.

- **X**: array-like of shape (n_samples, n_features)
- **y**: array-like of shape (n_samples,)
- **Returns**: self

#### `predict(X)`

Predict class labels.

- **X**: array-like of shape (n_samples, n_features)
- **Returns**: ndarray of shape (n_samples,)

#### `predict_proba(X)`

Predict class probabilities.

- **X**: array-like of shape (n_samples, n_features)
- **Returns**: ndarray of shape (n_samples, n_classes)

#### `predict_explain(X)`

Predict with full explainability output.

- **X**: array-like of shape (n_samples, n_features)
- **Returns**: [ZMatrix](#zmatrix)

#### `score(X, y, sample_weight=None)`

Return mean accuracy on the given test data.

- **X**: array-like of shape (n_samples, n_features)
- **y**: array-like of shape (n_samples,)
- **Returns**: float

### Attributes (after fitting)

| Attribute              | Type    | Description            |
|------------------------|---------|------------------------|
| `classes_`             | ndarray | The class labels       |
| `n_classes_`           | int     | Number of classes      |
| `n_features_in_`       | int     | Number of features     |
| `pattern_importances_` | ndarray | Feature importance scores (available after `predict_explain` is called) |

---

## ZMatrix

```python
from nsbc import ZMatrix
```

Dataclass returned by `predict_explain()`. Contains all information needed to explain predictions.

### Attributes

| Attribute              | Shape                        | Description                                          |
|------------------------|------------------------------|------------------------------------------------------|
| `z`                    | (n_test, n_train)            | Full Hamming similarity matrix                       |
| `class_scores`         | (n_test, n_classes)          | Sum of top-*u* similarities per class                |
| `predictions`          | (n_test,)                    | Predicted class labels                               |
| `top_u_indices`        | list of dict                 | Per test sample, maps class label to top-*u* training indices |
| `feature_importances`  | (n_test, n_features)         | Per-sample feature importance (mean match ratio across top-*u* neighbors) |
| `feature_bit_widths`   | (n_features,)                | Number of bits per feature in the binary encoding    |
| `classes`              | (n_classes,)                 | Class labels                                         |
| `n_value`              | int                          | The *u* parameter used                               |
| `x_test_encoded`       | (n_test, total_bit_width)    | Binary-encoded test samples                          |
| `x_train_encoded`      | (n_train, total_bit_width)   | Binary-encoded training samples                      |

### Properties

#### `global_feature_importances`

Returns the mean of `feature_importances` across all test samples. Shape: (n_features,).

---

## Plotting functions

```python
from nsbc.tools import (
    plot_feature_importances,
    plot_z_scores,
    plot_similarity_heatmap,
    plot_chunk_similarity,
)
```

!!! note
    Plotting functions require `matplotlib`. Install with `pip install nsbc[viz]`.

### `plot_feature_importances(z_matrix, sample_idx=None, feature_names=None, ax=None)`

Horizontal bar chart of feature importances.

- **sample_idx**: int or None. If None, plots global importances.
- **Returns**: (fig, ax)

### `plot_z_scores(z_matrix, sample_idx, y_train=None, top_k=10, ax=None)`

Bar chart of top-k Z-scores per class, grouped and sorted by similarity.

- **y_train**: training labels for coloring bars by class.
- **top_k**: number of top training samples to show per class.
- **Returns**: (fig, ax)

### `plot_similarity_heatmap(z_matrix, sample_idx, feature_names=None, top_k=None, ax=None)`

Heatmap of per-feature match ratios between a test sample and its top-*u* neighbors.

- **Returns**: (fig, ax)

### `plot_chunk_similarity(z_matrix, sample_idx, y_train, top_k=10, feature_names=None, ax=None)`

Two-panel plot: chunked bit-match ratios (left) and z-scores with arrows (right).

- **Returns**: (fig, (ax1, ax2))

---

## Utility functions

### `compute_feature_match_ratios(test_bits, train_bits, feature_bit_widths)`

```python
from nsbc.tools import compute_feature_match_ratios
```

Compute per-feature match ratios between a test sample and training samples.

- **test_bits**: ndarray of shape (total_bit_width,)
- **train_bits**: ndarray of shape (n_samples, total_bit_width)
- **feature_bit_widths**: ndarray of shape (n_features,)
- **Returns**: ndarray of shape (n_samples, n_features) with values in [0, 1]

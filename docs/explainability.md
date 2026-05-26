# Explainability

n-SBC provides built-in explainability through the `predict_explain()` method. Unlike post-hoc methods (e.g., SHAP), n-SBC's explanations are exact: the similarity scores and feature match ratios are the actual quantities used to make the prediction.

## Getting started

```python
import numpy as np
from nsbc import NSBCClassifier

clf = NSBCClassifier(n_value=3, decimals=2)
clf.fit(X_train, y_train)

# predict_explain returns a ZMatrix with full transparency
result = clf.predict_explain(X_test)
```

## The ZMatrix object

`result` is a `ZMatrix` containing:

| Attribute             | What it tells you                                  |
|-----------------------|----------------------------------------------------|
| `result.predictions`  | The predicted class for each test sample           |
| `result.z`            | Full similarity matrix (n_test x n_train)          |
| `result.class_scores` | Aggregated top-*u* scores per class                |
| `result.top_u_indices`| Which training samples were most similar per class |
| `result.feature_importances` | Per-feature match ratios for each prediction |

## Per-sample feature importances

Feature importances are computed as the mean bit-match ratio between a test sample and its top-*u* neighbors, decomposed by feature boundaries in the binary encoding.

```python
# Importances for the first test sample
imp = result.feature_importances[0]

# Rank features by importance
order = np.argsort(-imp)
for i in order:
    print(f"{feature_names[i]}: {imp[i]:.4f}")
```

## Global feature importances

Aggregate across all test samples to get a global view:

```python
global_imp = result.global_feature_importances
order = np.argsort(-global_imp)
for i in order:
    print(f"{feature_names[i]}: {global_imp[i]:.4f}")
```

## Finding the most similar training samples

```python
sample_idx = 0
pred_class = result.predictions[sample_idx]

# Indices of the top-u most similar training samples for the predicted class
top_indices = result.top_u_indices[sample_idx][pred_class]
print(f"Most similar training samples: {top_indices}")
print(f"Their similarity scores: {result.z[sample_idx, top_indices]}")
```

## Visualization

### Z-scores per class

Shows which training samples are most similar, grouped and colored by class:

```python
from nsbc.tools import plot_z_scores

fig, ax = plot_z_scores(result, sample_idx=0, y_train=y_train, top_k=10)
```

### Feature importances bar chart

```python
from nsbc.tools import plot_feature_importances

# Local (single sample)
fig, ax = plot_feature_importances(result, sample_idx=0, feature_names=feature_names)

# Global (all test samples)
fig, ax = plot_feature_importances(result, feature_names=feature_names)
```

### Similarity heatmap

Per-feature match ratios between a test sample and its top-*u* neighbors:

```python
from nsbc.tools import plot_similarity_heatmap

fig, ax = plot_similarity_heatmap(result, sample_idx=0, feature_names=feature_names)
```

### Chunk similarity (two-panel)

Left panel shows per-feature match ratios with color intensity; right panel shows total z-scores with arrows on top-*u* selections:

```python
from nsbc.tools import plot_chunk_similarity

fig, (ax1, ax2) = plot_chunk_similarity(
    result, sample_idx=0, y_train=y_train, top_k=10, feature_names=feature_names
)
```

## How it works

1. **Binary encoding**: Each feature is normalized, rounded, scaled, and converted to a Gray-coded binary string. The number of bits per feature depends on the maximum value after normalization.

2. **Hamming similarity**: For each test sample, the similarity to every training sample is the number of matching bits in their binary representations.

3. **Top-*u* selection**: For each class, the *u* training samples with the highest similarity are selected.

4. **Feature decomposition**: The binary string is split back at feature boundaries. The match ratio per feature is computed as the fraction of matching bits within that feature's segment, averaged across the top-*u* neighbors. This produces a value in [0, 1] for each feature.

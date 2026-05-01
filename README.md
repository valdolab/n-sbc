<p align="center">
  <img src="docs/nsbc.png" alt="n-SBC" width="400">
</p>

<p align="center">
  <a href="https://pypi.org/project/nsbc/"><img src="https://img.shields.io/pypi/v/nsbc" alt="PyPI"></a>
  <a href="https://pypi.org/project/nsbc/"><img src="https://img.shields.io/pypi/pyversions/nsbc" alt="Python"></a>
  <a href="https://github.com/valdolab/n-sbc/blob/main/LICENSE"><img src="https://img.shields.io/badge/license-MIT-blue" alt="License"></a>
  <a href="https://doi.org/10.3389/frai.2025.1610856"><img src="https://img.shields.io/badge/DOI-10.3389%2Ffrai.2025.1610856-blue" alt="Paper"></a>
  <a href="https://valdolab.github.io/n-sbc/"><img src="https://img.shields.io/badge/docs-online-green" alt="Docs"></a>
</p>

# n-SBC

A novel machine learning classifier based on Hamming similarity over Gray-coded binary representations. Scikit-learn compatible. n-SBC is a lazy learner: it stores the entire training set encoded as Gray-coded binary vectors. At prediction time, it computes the Hamming similarity between a new sample and every training sample, sums the top-*u* similarities per class, and predicts the class with the highest aggregate similarity. The Gray code encoding ensures that numerically close values differ by only one bit, preserving ordinal relationships in the binary representation.

> Velazquez-Gonzalez, O., Alarcon-Paredes, A., & Yanez-Marquez, C. (2026).
> *Medical pattern classification using a novel binary similarity approach based on an associative classifier.*
> Frontiers in Artificial Intelligence, 8. [DOI: 10.3389/frai.2025.1610856](https://doi.org/10.3389/frai.2025.1610856)

## Installation

```bash
pip install nsbc
```

## Quick Start

```python
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from nsbc import NSBCClassifier

X, y = load_iris(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

clf = NSBCClassifier(n_value=3, decimals=2, factor=10)
clf.fit(X_train, y_train)

print(f"Accuracy: {clf.score(X_test, y_test):.2%}")
```

## Parameters

| Parameter  | Type  | Default | Description                                      |
|------------|-------|---------|--------------------------------------------------|
| `n_value`  | int   | 3       | Number of top-u similar samples per class         |
| `decimals` | int   | 2       | Decimal places for rounding during normalization  |
| `factor`   | int   | 10      | Multiplicative factor applied after rounding      |

## Explainability

`predict_explain()` returns a `ZMatrix` with the full similarity matrix, per-class scores, and per-feature importances:

```python
result = clf.predict_explain(X_test)

# Feature importances for a single prediction
imp = result.feature_importances[0]
order = np.argsort(-imp)
for i in order:
    print(f"{feature_names[i]}: {imp[i]:.4f}")

result.global_feature_importances
```

Visualize which training samples are most similar and why:

```python
from nsbc.tools import plot_z_scores, plot_feature_importances

fig, ax = plot_z_scores(result, sample_idx=0, y_train=y_train)

# Global feature importances
fig, ax = plot_feature_importances(result, feature_names=feature_names)
```

## Examples

- [Basic usage](examples/01_basic_usage.ipynb): train, predict, evaluate with LOOCV
- [Explainability](examples/02_explainability.ipynb): Z-matrix, feature importances, similarity plots


## Citation

If you use n-SBC in your research, please cite:

```bibtex
@article{velazquez2026nsbc,
  title={Medical pattern classification using a novel binary similarity approach based on an associative classifier},
  author={Velazquez-Gonzalez, Osvaldo and Alarc{\'o}n-Paredes, Antonio and Ya{\~n}ez-Marquez, Cornelio},
  journal={Frontiers in Artificial Intelligence},
  volume={8},
  year={2026},
  month={1},
  doi={10.3389/frai.2025.1610856}
}
```

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines on reporting bugs, suggesting features, and submitting pull requests.

## License

MIT -- see [LICENSE](LICENSE) for details.

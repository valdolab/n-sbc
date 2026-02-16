# n-SBC

A lazy machine learning classifier based on Hamming similarity over Gray-coded binary representations. Scikit-learn compatible.

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

## Examples

See the `examples/` directory for usage notebooks. Future releases will include explainability and visualization notebooks.

## How It Works

n-SBC is a lazy learner: it stores the entire training set encoded as Gray-coded binary vectors. At prediction time, it computes the Hamming similarity between a new sample and every training sample, sums the top-*u* similarities per class, and predicts the class with the highest aggregate similarity. The Gray code encoding ensures that numerically close values differ by only one bit, preserving ordinal relationships in the binary representation.

## Citation

If you use n-SBC in your research, please cite:

```bibtex
@article{velazquez2026medical,
  title={Medical pattern classification using a novel binary similarity approach based on an associative classifier},
  author={Velazquez-Gonzalez, Osvaldo and Alarc{\'o}n-Paredes, Antonio and Ya{\~n}ez-Marquez, Cornelio},
  journal={Frontiers in Artificial Intelligence},
  volume={8},
  year={2026},
  month={1},
  doi={10.3389/frai.2025.1610856}
}
```

## License

MIT -- see [LICENSE](LICENSE) for details.

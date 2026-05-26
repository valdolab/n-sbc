# Quick Start

## Installation

```bash
pip install nsbc
```

For plotting capabilities:

```bash
pip install nsbc[viz]
```

## Basic usage

```python
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from nsbc import NSBCClassifier

X, y = load_iris(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

clf = NSBCClassifier(n_value=3, decimals=2, factor=10)
clf.fit(X_train, y_train)

print(f"Accuracy: {clf.score(X_test, y_test):.2%}")
```

## Using with scikit-learn pipelines

```python
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

pipe = Pipeline([
    ("scaler", StandardScaler()),
    ("clf", NSBCClassifier(n_value=3, decimals=2)),
])
pipe.fit(X_train, y_train)
print(f"Pipeline accuracy: {pipe.score(X_test, y_test):.2%}")
```

## Cross-validation

```python
from sklearn.model_selection import cross_val_score

scores = cross_val_score(
    NSBCClassifier(n_value=3, decimals=2),
    X, y, cv=5, scoring="accuracy"
)
print(f"CV accuracy: {scores.mean():.2%} (+/- {scores.std():.2%})")
```

## Parameters

| Parameter  | Type | Default | Description                                     |
|------------|------|---------|-------------------------------------------------|
| `n_value`  | int  | 3       | Number of top-*u* similar samples per class     |
| `decimals` | int  | 2       | Decimal places for rounding during normalization|
| `factor`   | int  | 10      | Multiplicative factor applied after rounding    |

## Next steps

- [Explainability guide](explainability.md) — understand predictions with feature importances and similarity plots
- [API Reference](api.md) — full method documentation

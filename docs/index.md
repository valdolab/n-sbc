# n-SBC

**N-Similarity Binary Classifier** — a lazy ML classifier based on Hamming similarity over Gray-coded binary representations.

[![PyPI](https://img.shields.io/pypi/v/nsbc)](https://pypi.org/project/nsbc/)
[![Python](https://img.shields.io/pypi/pyversions/nsbc)](https://pypi.org/project/nsbc/)
[![License](https://img.shields.io/badge/license-MIT-blue)](https://github.com/valdolab/n-sbc/blob/main/LICENSE)
[![DOI](https://zenodo.org/badge/1059062228.svg)](https://doi.org/10.5281/zenodo.20545100)

## Overview

n-SBC is a lazy machine learning model that stores the entire training set encoded as Gray-coded binary vectors. The Gray code encoding ensures that numerically close values differ by only one bit, preserving ordinal relationships in the binary representation.

## Key features

- **scikit-learn compatible**: `fit`, `predict`, `predict_proba`, `score`, pipelines, cross-validation
- **Built-in explainability**: `predict_explain()` returns similarity matrices and per-feature importances without external libraries
- **Visualization tools**: publication-ready plots for similarity scores, feature importances, and pattern matching
- **Lightweight**: depends only on NumPy, scikit-learn, and tqdm

## Installation

```bash
pip install nsbc
```

For visualization support:

```bash
pip install nsbc[viz]
```

## Citation

If you use n-SBC in your research, please cite:

> Velazquez-Gonzalez, O., Alarcon-Paredes, A., & Yanez-Marquez, C. (2026).
> *Medical pattern classification using a novel binary similarity approach based on an associative classifier.*
> Frontiers in Artificial Intelligence, 8. DOI: [10.3389/frai.2025.1610856](https://doi.org/10.3389/frai.2025.1610856)

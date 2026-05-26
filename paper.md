---
title: 'n-SBC: n-Similarity Binary Classfier, a Python package for explainable machine learning classification'
tags:
  - Python
  - machine learning
  - classification
  - explainability
  - Gray code
  - Hamming similarity
authors:
  - name: Osvaldo Velazquez-Gonzalez
    orcid: 0009-0004-8935-6288
    affiliation: 1
affiliations:
  - name: Centro de Investigacion en Computacion, Instituto Politecnico Nacional, Mexico
    index: 1
date: 20 April 2026
bibliography: paper.bib
---

# Summary

`n-SBC` is a Python package that implements the N-Similarity Binary Classifier. This is a lazy machine learning algorithm for pattern classification. It computes Hamming similarity over Gray coded binary strings. The package includes a class that works with scikit-learn, making it easy to add to existing machine learning workflows. It also has built-in explainability tools that show which training or testing samples and features affect each model prediction.

The classifier encodes numerical features into Gray code binary vectors. In this system, numerically close samples only differ by one bit. It measures similarity between a test sample and all stored training samples through bitwise comparison. Predictions happen by summing the top-*u* similarity scores for each class. The class with the highest total score is selected. This method naturally produces clear similarity scores, which can be broken down at the feature level.

# Statement of need

The n-SBC algorithm was introduced in @velazquez2026medical as a novel machine learning model for pattern classification. It showed competitive results on medical datasets. However, the original version was created in MATLAB, which limited access for the wider machine learning community [@pedregosa2011scikit]. There was no open-source version available that could be installed with pip. This lack of access made it hard for other researchers to reproduce the published results and adopt the method.

`n-SBC` provides a Python package that works with scikit-learn. In addition, the current packeage offers visualization tools that clarify predictions at both the feature and sample levels,  which is valuable for researchers who need interpretable classifiers in fields like medical diagnosis, where understanding the reasons behind a prediction is just as important as the prediction itself.

# State of the field

No existing software package implements the n-SBC algorithm. The closest alternatives are packages for similar instance based models. Distance based classifiers like k-nearest neighbors (kNN), available in scikit-learn [@pedregosa2011scikit], are popular because they are simple and easy to understand. However, standard kNN uses Euclidean or Minkowski distances based on raw feature values and does not directly provide feature-level explanations. Post-hoc explainability methods, such as SHAP [@lundberg2017unified], can work with any model but give only approximate explanations that do not fully capture how the model computes results. n-SBC is the first open-source implementation of this algorithm. It offers built-in explainability where feature contributions come directly from the prediction process instead of being an external approximation.

# Software design

The package architecture consists of three layers:

1. **NSBCClassifier** (`estimator.py`): The user-facing estimator following scikit-learn conventions. Accepts parameters `n_value` (number of top neighbors per class), `decimals` (rounding precision), and `factor` (multiplicative scaling). Inherits from `sklearn.base.BaseEstimator` and `ClassifierMixin`.
2. **NSBCEngine** (`engine.py`): The core algorithm that normalizes input data, encodes features into Gray-coded binary vectors using vectorized NumPy [@harris2020array] operations, stores the training set, and computes Hamming similarities at prediction time.
3. **Tools module** (`tools/`): Contains the `ZMatrix` dataclass for structured explainability output, `compute_feature_match_ratios` for per-feature decomposition, and four plotting functions for visualization.

The normalization pipeline replicates the original implementation exactly [@velazquez2026medical], ensuring numerical equivalence.

A typical workflow:

```python
from nsbc import NSBCClassifier

clf = NSBCClassifier(n_value=3, decimals=2)
clf.fit(X_train, y_train)
result = clf.predict_explain(X_test)

# Per-sample feature importances
result.feature_importances

result.global_feature_importances
```

The `ZMatrix` object provides full transparency: `result.z` contains the raw similarity matrix, `result.top_u_indices` maps each class to its most similar training samples, and `result.feature_importances` quantifies each feature's contribution by computing match ratios within the binary encoding boundaries.

# Research impact statement

The n-SBC algorithm has been validated on medical datasets including cryotherapy and immunotherapy treatment outcomes [@velazquez2026medical], demonstrating competitive classification performance. The Python package enables reproducibility of these results—the included test suite verifies numerical equivalence with the original MATLAB implementation.

# AI usage disclosure

Generative AI (Claude Opus 4.6 from Anthropic) assisted with implementations of code tests , and documentation. All outputs were reviewed, tested, and validated by the author. The scientific content, algorithmic design, and research contributions are entirely the work of the authors as described in @velazquez2026medical.

# Acknowledgements

The author acknowledges the original theoretical development of the n-SBC algorithm by Velazquez-Gonzalez, Alarcon-Paredes, and Yanez-Marquez.

# References

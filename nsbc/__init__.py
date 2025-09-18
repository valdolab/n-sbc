"""
n-SBC: A novel machine learning model compatible with scikit-learn
"""

from .__version__ import __version__
from .estimator import NSBCClassifier

__all__ = [
    "__version__",
    "NSBCClassifier",
]

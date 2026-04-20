"""
n-SBC: A novel machine learning model compatible with scikit-learn
"""

from .__version__ import __version__
from .estimator import NSBCClassifier
from .tools.matrix_z import ZMatrix

__all__ = [
    "__version__",
    "NSBCClassifier",
    "ZMatrix",
]

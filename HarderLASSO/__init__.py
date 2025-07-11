"""
HarderLASSO: Neural Network Feature Selection Library

HarderLASSO is a Python library for neural network-based feature selection
using regularized optimization techniques. It provides scikit-learn compatible
estimators for various tasks including regression, classification, and survival analysis.

Key Features:
- Multi-phase training with adaptive regularization
- Quantile Universal Threshold (QUT) for automatic lambda selection
- Support for various penalty functions (LASSO, Harder, SCAD)
- Scikit-learn compatible API
- Comprehensive visualization tools

Examples:
    >>> from harderLASSO import HarderLASSORegressor
    >>> import numpy as np
    >>> X = np.random.randn(100, 20)
    >>> y = np.random.randn(100)
    >>> model = HarderLASSORegressor(hidden_dims=(10,))
    >>> model.fit(X, y)
    >>> predictions = model.predict(X)
    >>> print(f"Selected {len(model.selected_features_)} features")
"""

# Main estimators - public API
from .regressor import HarderLASSORegressor
from .classifier import HarderLASSOClassifier
from .cox import HarderLASSOCox
import HarderLASSO._utils._visuals

__version__ = "0.1.0"
__author__ = "HarderLASSO Team"
__email__ = "contact@harderlasso.org"

__all__ = [
    'HarderLASSORegressor',
    'HarderLASSOClassifier',
    'HarderLASSOCox',
]

"""HarderLASSO classifier for neural network feature selection.

This module provides the HarderLASSOClassifier, which combines neural network
learning with feature selection using the HarderLASSO regularization method
for classification tasks.
"""

from typing import Optional, Tuple
from torch import nn
from ._base import _BaseHarderLASSOModel
from ._tasks import _ClassificationTaskMixin
from ._qut import _ClassificationQUT


class HarderLASSOClassifier(_BaseHarderLASSOModel, _ClassificationTaskMixin):
    """Neural network classifier with automatic feature selection using HarderLASSO.

    HarderLASSOClassifier combines neural network learning with feature selection
    using the HarderLASSO regularization method for classification tasks. It
    automatically identifies the most relevant features while training a neural
    network classifier, making it particularly suitable for high-dimensional
    classification problems where interpretability is important.

    The training process uses a multi-phase optimization strategy:

    1. **Initialization Phase**: Unregularized optimization to establish good
       initial parameter values
    2. **Regularization Phase**: Gradual introduction of sparsity-inducing penalties
    3. **Refinement Phase**: FISTA optimization for precise sparse solutions

    The regularization parameter is automatically determined using the Quantile
    Universal Threshold (QUT) method, eliminating the need for manual tuning.

    Parameters
    ----------
    hidden_dims : tuple of int, default=(20,)
        Number of neurons in each hidden layer. If None, creates a linear model.
    lambda_qut : float, optional
        Regularization parameter for sparsity control. If None, computed
        automatically using QUT.
    penalty : {'harder', 'lasso', 'scad'}, default='harder'
        Type of penalty function for regularization.
    activation : nn.Module, optional
        Activation function to use in hidden layers. Defaults to nn.ReLU().

    Attributes
    ----------
    coef_ : ndarray
        Regression coefficients of the linear output layer after fitting.
    selected_features_ : list
        Names of features selected by the model.
    feature_names_in_ : ndarray
        Names of features seen during fit.
    n_features_in_ : int
        Number of features seen during fit.
    lambda_qut_ : float
        The computed or provided regularization parameter.
    selected_features_indices_ : ndarray
        Indices of selected features.

    Examples
    --------
    >>> from sklearn.datasets import load_digits
    >>> from HarderLASSO import HarderLASSOClassifier
    >>>
    >>> X, y = load_digits(return_X_y=True)
    >>> model = HarderLASSOClassifier(hidden_dims=(20,))
    >>> model.fit(X, y)
    >>> print(f"Selected features: {len(model.selected_features_)}")

    Notes
    -----
    - Input features are automatically standardized
    - Supports both numpy arrays and pandas DataFrames
    - Handles binary and multi-class classification

    See Also
    --------
    HarderLASSORegressor : For regression tasks
    HarderLASSOCox : For survival analysis tasks
    """

    def __init__(
        self,
        hidden_dims: Tuple[int, ...] = (20,),
        lambda_qut: Optional[float] = None,
        penalty: str = 'harder',
        activation: Optional[nn.Module] = nn.ReLU(),
        alpha: float = 0.05
    ) -> None:
        """Initialize the HarderLASSO classifier.

        Parameters
        ----------
        hidden_dims : tuple of int, default=(20,)
            Sizes of hidden layers in the neural network.
        lambda_qut : float, optional
            Regularization parameter for feature selection.
        penalty : {'harder', 'lasso', 'ridge'}, default='harder'
            Type of regularization to apply.
        activation : nn.Module, optional
            Activation function to use in hidden layers. Defaults to nn.ReLU().
        alpha: float, default=0.05
            Significance level for QUT computation.
        """
        # Initialize base model
        _BaseHarderLASSOModel.__init__(
            self,
            hidden_dims=hidden_dims,
            output_dim=None,  # Will be set automatically based on data
            bias=True,
            penalty=penalty,
            activation=activation
        )

        # Initialize task-specific functionality
        _ClassificationTaskMixin.__init__(self)

        # Initialize QUT manager
        self.QUT = _ClassificationQUT(lambda_qut=lambda_qut, alpha=alpha)

    def __repr__(self) -> str:
        """Return string representation of the classifier."""
        return (
            f"HarderLASSOClassifier("
            f"hidden_dims={self._neural_network.hidden_dims}, "
            f"lambda_qut={self.lambda_qut_}, "
            f"penalty='{self.penalty}', "
        )

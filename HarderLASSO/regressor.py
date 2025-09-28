"""HarderLASSO Regressor for sparse neural network regression.

This module provides the HarderLASSORegressor class, which implements
neural network-based regression with automatic feature selection using
the HarderLASSO methodology. The estimator follows scikit-learn conventions
and provides both linear and nonlinear regression capabilities with
integrated feature selection.
"""

import numpy as np
from torch import nn
from typing import Optional, Tuple, Union, List, Any
from ._base import _BaseHarderLASSOModel
from ._tasks import _RegressionTaskMixin
from ._qut import _RegressionQUT


class HarderLASSORegressor(_BaseHarderLASSOModel, _RegressionTaskMixin):
    """Neural network regressor with automatic feature selection using HarderLASSO.

    HarderLASSORegressor implements a neural network-based regression model that
    automatically selects relevant features using regularized optimization. The
    model combines the expressiveness of neural networks with the interpretability
    of feature selection, making it suitable for high-dimensional regression tasks
    where understanding which features are important is crucial.

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
    >>> from HarderLASSO import HarderLASSORegressor
    >>> import numpy as np
    >>>
    >>> X = np.random.randn(100, 50)
    >>> y = X[:, [0, 2, 5]].sum(axis=1) + np.random.randn(100)
    >>> model = HarderLASSORegressor(hidden_dims=(10,))
    >>> model.fit(X, y)
    >>> print(f"Selected features: {len(model.selected_features_)}")

    Notes
    -----
    - Input features are automatically standardized
    - Supports both numpy arrays and pandas DataFrames

    See Also
    --------
    HarderLASSOClassifier : For classification tasks
    HarderLASSOCox : For survival analysis tasks
    """

    def __init__(
        self,
        hidden_dims: Optional[Tuple[int, ...]] = (20,),
        lambda_qut: Optional[float] = None,
        penalty: str = 'harder',
        activation: Optional[nn.Module] = nn.ReLU(),
        alpha: float = 0.05
    ):
        """Initialize the HarderLASSO regressor.

        Parameters
        ----------
        hidden_dims : tuple of int, default=(20,)
            Architecture of the neural network hidden layers.
        lambda_qut : float, optional
            Regularization strength parameter.
        penalty : {'harder', 'lasso', 'scad'}, default='harder'
            Type of regularization penalty to apply.
        activation : nn.Module, optional
            Activation function for the hidden layers. Defaults to nn.ReLU().
        alpha: float, default=0.05
            Significance level for QUT computation.
        """
        # Initialize base model
        _BaseHarderLASSOModel.__init__(
            self,
            hidden_dims=hidden_dims,
            output_dim=1,
            bias=True,
            penalty=penalty,
            activation=activation
        )

        # Initialize task-specific functionality
        _RegressionTaskMixin.__init__(self)

        # Initialize QUT component
        self.QUT = _RegressionQUT(lambda_qut=lambda_qut, alpha=alpha)
        self.lambda_qut_ = lambda_qut

    def plot_diagnostics(
        self,
        X: Union[np.ndarray, Any],
        y: Union[np.ndarray, Any]
    ) -> None:
        """Plot comprehensive regression diagnostics.

        Creates a 2x2 diagnostic plot layout including actual vs predicted,
        residuals vs predicted, residual distribution, and Q-Q plot.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Input features for generating predictions.
        y : array-like of shape (n_samples,)
            True target values.

        Examples
        --------
        >>> model.fit(X_train, y_train)
        >>> model.plot_diagnostics(X_test, y_test)
        """
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        fig.suptitle('HarderLASSO Regression Diagnostics', fontsize=16)

        self.plot_actual_vs_predicted(X, y, ax=axes[0, 0])
        self.plot_residuals_vs_predicted(X, y, ax=axes[0, 1])
        self.plot_residual_distribution(X, y, ax=axes[1, 0])
        self.plot_qq(X, y, ax=axes[1, 1])

        plt.tight_layout()
        plt.show()

    def plot_actual_vs_predicted(
        self,
        X: Union[np.ndarray, Any],
        y: Union[np.ndarray, Any],
        ax=None,
        **plot_kwargs
    ) -> None:
        """Plot actual vs predicted values.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Input features for generating predictions.
        y : array-like of shape (n_samples,)
            True target values.
        ax : matplotlib.axes.Axes, optional
            Matplotlib axes object for plotting.
        **plot_kwargs
            Additional keyword arguments for plotting customization.
        """
        from ._utils._visuals import _plot_actual_vs_predicted

        y_pred = self.predict(X)
        _plot_actual_vs_predicted(y_pred, y, ax=ax, **plot_kwargs)

    def plot_residuals_vs_predicted(
        self,
        X: Union[np.ndarray, Any],
        y: Union[np.ndarray, Any],
        ax=None,
        **plot_kwargs
    ) -> None:
        """Plot residuals vs predicted values.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Input features for generating predictions.
        y : array-like of shape (n_samples,)
            True target values.
        ax : matplotlib.axes.Axes, optional
            Matplotlib axes object for plotting.
        **plot_kwargs
            Additional keyword arguments for plotting customization.
        """
        from ._utils._visuals import _plot_residuals_vs_predicted

        y_pred = self.predict(X)
        _plot_residuals_vs_predicted(y_pred, y, ax=ax, **plot_kwargs)

    def plot_residual_distribution(
        self,
        X: Union[np.ndarray, Any],
        y: Union[np.ndarray, Any],
        ax=None,
        **plot_kwargs
    ) -> None:
        """Plot histogram of residuals for normality assessment.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Input features for generating predictions.
        y : array-like of shape (n_samples,)
            True target values.
        ax : matplotlib.axes.Axes, optional
            Matplotlib axes object for plotting.
        **plot_kwargs
            Additional keyword arguments for plotting customization.
        """
        from ._utils._visuals import _plot_residual_distribution

        y_pred = self.predict(X)
        _plot_residual_distribution(y_pred, y, ax=ax, **plot_kwargs)

    def plot_qq(
        self,
        X: Union[np.ndarray, Any],
        y: Union[np.ndarray, Any],
        ax=None,
        **plot_kwargs
    ) -> None:
        """Plot Q-Q plot for residual normality assessment.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Input features for generating predictions.
        y : array-like of shape (n_samples,)
            True target values.
        ax : matplotlib.axes.Axes, optional
            Matplotlib axes object for plotting.
        **plot_kwargs
            Additional keyword arguments for plotting customization.
        """
        from ._utils._visuals import _plot_qq

        y_pred = self.predict(X)
        _plot_qq(y_pred, y, ax=ax, **plot_kwargs)

    def get_feature_importance(self) -> dict:
        """Get feature importance scores based on model coefficients.

        Returns
        -------
        dict
            Dictionary mapping feature names to importance scores.

        Examples
        --------
        >>> model.fit(X, y)
        >>> importance = model.get_feature_importance()
        >>> top_features = sorted(importance.items(), key=lambda x: x[1], reverse=True)[:5]
        """
        return super().get_feature_importance()

    def __repr__(self) -> str:
        """String representation of the estimator."""
        return (
            f"HarderLASSORegressor("
            f"hidden_dims={self._neural_network._hidden_dims}, "
            f"lambda_qut={self.lambda_qut_}, "
            f"penalty='{self.penalty}')"
        )

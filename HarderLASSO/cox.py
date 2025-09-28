"""HarderLASSO Cox regression for neural network feature selection.

This module provides the HarderLASSOCox estimator, which combines neural network
learning with feature selection using the HarderLASSO regularization method
for survival analysis tasks.
"""

from typing import Optional, Tuple, Union, List
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from numpy import ndarray

from ._base import _BaseHarderLASSOModel
from ._tasks import _CoxTaskMixin
from ._qut import _CoxQUT
from ._utils import survival_analysis_training_info


class HarderLASSOCox(_BaseHarderLASSOModel, _CoxTaskMixin):
    """Neural network Cox model with automatic feature selection using HarderLASSO.

    HarderLASSOCox combines neural network learning with feature selection using
    the HarderLASSO regularization method for survival analysis. It implements
    the Cox proportional hazards model with neural network-based feature selection,
    making it suitable for high-dimensional survival data where identifying
    prognostic factors is crucial.

    The estimator extends the traditional Cox model by:

    1. **Neural Network Enhancement**: Uses neural networks to capture complex
       nonlinear relationships between features and hazard rates
    2. **Automatic Feature Selection**: Applies structured sparsity to identify
       the most important prognostic features
    3. **Regularization**: Uses QUT-based automatic parameter selection to
       avoid overfitting in high-dimensional settings

    The model maintains the interpretability of Cox regression while adding
    the expressiveness of neural networks and the benefits of automatic
    feature selection.

    The training process uses a multi-phase optimization strategy:

    1. **Initialization Phase**: Unregularized optimization to establish good
       initial parameter values
    2. **Regularization Phase**: Gradual introduction of sparsity-inducing penalties
    3. **Refinement Phase**: FISTA optimization for precise sparse solutions

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
        Activation function to use in the neural network. Defaults to ReLU.
    approximation : {'Breslow', 'Efron'}, default='Breslow'
        Method for handling tied event times.

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
    baseline_hazard_ : DataFrame
        Estimated baseline cumulative hazard function.
    baseline_survival_ : DataFrame
        Estimated baseline survival function.
    survival_functions_ : DataFrame
        Individual survival functions for training samples.
    training_survival_df_ : DataFrame
        Comprehensive survival analysis results.

    Examples
    --------
    >>> from HarderLASSO import HarderLASSOCox
    >>> import pandas as pd
    >>> # Assume df has columns: time, event, feature1, feature2, ...
    >>> X = df.drop(['time', 'event'], axis=1)
    >>> y = df[['time', 'event']].values
    >>> cox = HarderLASSOCox()
    >>> cox.fit(X, y)
    >>> print(f"Selected features: {len(cox.selected_features_)}")

    Notes
    -----
    - Input features are automatically standardized
    - Supports both numpy arrays and pandas DataFrames
    - Assumes proportional hazards assumption holds
    - Survival times and event indicators must be provided as a 2D array
    - The model assumes the proportional hazards assumption holds

    See Also
    --------
    HarderLASSORegressor : For regression tasks
    HarderLASSOClassifier : For classification tasks
    """

    def __init__(
        self,
        hidden_dims: Tuple[int, ...] = (20,),
        lambda_qut: Optional[float] = None,
        penalty: str = 'harder',
        activation: Optional[nn.Module] = nn.ReLU(),
        approximation: str = 'Breslow',
        alpha: float = 0.05
    ) -> None:
        """Initialize the HarderLASSO Cox regression model.

        Parameters
        ----------
        hidden_dims : tuple of int, default=(20,)
            Sizes of hidden layers in the neural network.
        lambda_qut : float, optional
            Regularization parameter for feature selection.
        penalty : {'harder', 'lasso', 'ridge'}, default='harder'
            Type of regularization to apply.
        activation : nn.Module, optional
            Activation function to use in the neural network. Defaults to ReLU.
        approximation : {'Breslow', 'Efron'}, default='Breslow'
            Method for handling tied event times.
        alpha: float, default=0.05
            Significance level for QUT computation.
        """
        # Initialize base model
        _BaseHarderLASSOModel.__init__(
            self,
            hidden_dims=hidden_dims,
            output_dim=1,
            bias=False,
            penalty=penalty,
            activation=activation
        )

        # Initialize task-specific functionality
        _CoxTaskMixin.__init__(self, approximation=approximation)

        # Initialize QUT manager
        self.QUT = _CoxQUT(
            lambda_qut=lambda_qut,
            alpha=alpha
        )
        self.lambda_qut_ = lambda_qut

    @property
    def baseline_hazard_(self) -> pd.DataFrame:
        """Baseline cumulative hazard function.

        Returns
        -------
        DataFrame
            DataFrame with index=durations and column=baseline_cumulative_hazard

        Raises
        ------
        ValueError
            If model has not been fitted yet.
        """
        if self.training_survival_df_ is None:
            raise ValueError("Model not trained yet.")
        df = self.training_survival_df_[["durations", "baseline_cumulative_hazard"]]
        df = df.set_index("durations", drop=True)
        return df

    @property
    def baseline_survival_(self) -> pd.DataFrame:
        """Baseline survival function.

        Returns
        -------
        DataFrame
            DataFrame with index=durations and column=baseline_survival

        Raises
        ------
        ValueError
            If model has not been fitted yet.
        """
        if self.training_survival_df_ is None:
            raise ValueError("Model not trained yet.")
        df = self.training_survival_df_[["durations", "baseline_survival"]]
        df = df.set_index("durations", drop=True)
        return df

    @property
    def survival_functions_(self) -> pd.DataFrame:
        """Individual survival functions for training samples.

        Returns
        -------
        DataFrame
            DataFrame with index=durations and columns=individual_i for each
            training sample i, containing S_i(t).

        Raises
        ------
        ValueError
            If model has not been fitted yet.
        """
        if self.training_survival_df_ is None:
            raise ValueError("Model not trained yet.")

        df = self.training_survival_df_
        individual_cols = [c for c in df.columns if c.startswith("individual_")]
        return df.loc[:, ["durations"] + individual_cols].set_index("durations")

    def fit(
        self,
        X: Union[ndarray, torch.Tensor],
        y: Union[ndarray, torch.Tensor],
        verbose: bool = False,
        logging_interval: int = 50,
        lambda_path: Optional[List[float]] = None
    ) -> 'HarderLASSOCox':
        """Fit the HarderLASSO Cox regression model to training data.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data features.
        y : array-like of shape (n_samples, 2)
            Training data with survival times and event indicators.
            First column contains survival times, second column contains
            event indicators (1=event, 0=censored).
        verbose: Whether to print training progress.
        logging_interval: Frequency of progress updates during training.
        lambda_path: Custom regularization path. If None, computed automatically.

        Returns
        -------
        self : HarderLASSOCox
            The fitted Cox regression model.

        Raises
        ------
        ValueError
            If input data has invalid format or shape.
        """
        # Call parent fit method
        super().fit(
            X, y,
            verbose = verbose,
            logging_interval = logging_interval,
            lambda_path = lambda_path
        )

        y = np.asarray(y, dtype=np.float32)
        if y.shape[0] == 2 and y.shape[1] != 2:
            y = y.T

        times = y[:, 0]
        events = y[:, 1]
        predictions = self.predict(X)

        self.training_survival_df_ = survival_analysis_training_info(
            times, events, predictions
        )

        return self

    def plot_diagnostics(
        self,
        X: Union[ndarray, pd.DataFrame],
    ) -> None:
        """Plot comprehensive survival diagnostics.

        Creates diagnostic plots including baseline hazard, baseline survival,
        and feature effects for selected features.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Covariate matrix used for computing feature effects.

        Examples
        --------
        >>> model.fit(X_train, y_train)
        >>> model.plot_diagnostics(X_test)
        """
        import matplotlib.pyplot as plt
        from math import ceil

        features = self.selected_features_
        s = len(features)
        n_rows = 1 + ceil((s) / 2)

        fig, axes = plt.subplots(n_rows, 2, figsize=(12, 4 * n_rows), squeeze=False)
        fig.suptitle('HarderLASSO Survival Diagnostics', fontsize=16)

        self.plot_baseline_hazard(ax=axes[0, 0])
        self.plot_baseline_survival(ax=axes[0, 1])

        # 4) Four feature‐effect plots in the remaining subplots
        features = self.selected_features_
        for ax, feat in zip(axes.ravel()[2:], features):
            self.plot_feature_effects(
                feature_name=feat,
                X=X,
                ax=ax,
                linewidth=2,
            )
            ax.set_title(f"Effect of {feat}")

        # 5) Tidy up
        for ax in axes.ravel():
            ax.set_xlabel("Time")
        fig.tight_layout()
        plt.show()

    def plot_baseline_hazard(self, ax=None, **plot_kwargs):
        """Plot the estimated baseline cumulative hazard function.

        Parameters
        ----------
        ax : matplotlib.axes.Axes, optional
            If provided, the plot will be drawn on this axes.
        **plot_kwargs
            Additional keyword arguments passed to matplotlib.axes.Axes.plot.
        """
        from ._utils._visuals._cox_plots import _plot_baseline_cumulative_hazard
        _plot_baseline_cumulative_hazard(self.baseline_hazard_, ax=ax, **plot_kwargs)

    def plot_baseline_survival(self, ax=None, **plot_kwargs):
        """Plot the estimated baseline survival function.

        Parameters
        ----------
        ax : matplotlib.axes.Axes, optional
            If provided, the plot will be drawn on this axes.
        **plot_kwargs
            Additional keyword arguments passed to matplotlib.axes.Axes.plot.
        """
        from ._utils._visuals._cox_plots import _plot_baseline_survival_function
        _plot_baseline_survival_function(self.baseline_survival_, ax=ax, **plot_kwargs)

    def plot_survival(
        self,
        predictions: Optional[ndarray] = None,
        individuals_idx: Optional[Union[int, List[int]]] = None,
        ax=None,
        **plot_kwargs
    ):
        """Plot individual survival curves.

        Parameters
        ----------
        predictions : array-like of shape (n_individuals,), optional
            Log-hazard predictions for each individual. If None,
            uses the precomputed training curves.
        individuals_idx : int or list of int, optional
            Index or list of indices of individuals to plot.
            If None, plots all individuals.
        ax : matplotlib.axes.Axes, optional
            If provided, the plot will be drawn on this axes.
        **plot_kwargs
            Additional keyword arguments passed to matplotlib.axes.Axes.plot.
        """
        from ._utils._visuals._cox_plots import _plot_survival_curves

        durations = self.baseline_survival_.index.values

        if predictions is not None:
            baseline = self.baseline_survival_["baseline_survival"].values
            preds = np.asarray(predictions)
            survival_matrix = baseline[:, None] ** np.exp(preds)
            labels = [f"individual_{i}" for i in range(len(preds))]
        else:
            sf_df = self.survival_functions_
            survival_matrix = sf_df.values
            labels = list(sf_df.columns)

        _plot_survival_curves(
            durations=durations,
            survival_matrix=survival_matrix,
            labels=labels,
            individuals_idx=individuals_idx,
            ax=ax,
            **plot_kwargs
        )

    def plot_feature_effects(
        self,
        feature_name: str,
        X: ndarray,
        values: Optional[List] = None,
        ax=None,
        **plot_kwargs
    ):
        """Plot how varying a feature affects survival curves.

        Parameters
        ----------
        feature_name : str
            The feature to vary.
        X : array-like of shape (n_samples, n_features)
            Covariate matrix to use for computing mean values.
        values : list of scalars, optional
            Values of the feature to plot. If None, uses quantiles.
        ax : matplotlib.axes.Axes, optional
            If provided, the plot will be drawn on this axes.
        **plot_kwargs
            Additional keyword arguments passed to the plot function.
        """
        df = self.training_survival_df_
        durations = df["durations"].values
        baseline_surv = df["baseline_survival"].values

        if feature_name not in self.feature_names_in_:
            raise ValueError(f"'{feature_name}' not in feature names")

        idx = self.feature_names_in_.index(feature_name)

        from ._utils._visuals._cox_plots import _plot_feature_effects

        _plot_feature_effects(
            durations=durations,
            baseline_survival=baseline_surv,
            X=X,
            predict_fn=self.predict,
            feature_name=feature_name,
            idx=idx,
            values=values,
            ax=ax,
            **plot_kwargs
        )

    def __repr__(self) -> str:
        """Return string representation of the Cox regression model."""
        return (
            f"HarderLASSOCox("
            f"hidden_dims={self._neural_network.hidden_dims}, "
            f"lambda_qut={self.lambda_qut_}, "
            f"penalty='{self.penalty}', "
            f"approximation='{self.approximation}', "
            f"bias=False)"
        )

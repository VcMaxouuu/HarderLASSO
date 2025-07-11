"""
Feature selection mixin for HarderLASSO models.

This module provides the _FeatureSelectionMixin class that implements
feature selection functionality through regularized neural network training.
"""

import torch
import numpy as np
import warnings
from typing import List, Optional, Tuple, Dict, Any, Union


class _FeatureSelectionMixin:
    """Mixin class providing feature selection functionality for neural networks.

    This mixin implements feature selection through regularized optimization
    of neural network weights. It supports various penalty functions and
    provides methods for pruning and analyzing selected features.

    Parameters
    ----------
    lambda_qut : float, optional
        Regularization parameter for controlling sparsity. If None, computed
        automatically.
    penalty : {'lasso', 'harder', 'scad'}, default='harder'
        Type of penalty to apply.

    Attributes
    ----------
    lambda_qut_ : float
        Regularization parameter value.
    penalty : str
        Type of penalty function used.
    selected_features_indices_ : list
        Indices of selected features after training.
    feature_names_in_ : list
        Names of input features.
    penalized_parameters_ : list
        Parameters subject to regularization.
    unpenalized_parameters_ : list
        Parameters not subject to regularization.

    Notes
    -----
    This is an internal mixin class and should not be used directly.
    Use the public HarderLASSO classes instead.
    """

    def __init__(self, lambda_qut: Optional[float] = None, penalty: str = 'harder'):
        """Initialize the feature selection mixin.

        Parameters
        ----------
        lambda_qut : float, optional
            Regularization parameter for controlling sparsity.
        penalty : {'lasso', 'harder', 'scad'}, default='harder'
            Type of penalty to apply.
        """
        if penalty not in ['lasso', 'harder', 'scad']:
            raise ValueError(f"Penalty must be one of ['lasso', 'harder', 'scad'], got {penalty}")

        if not hasattr(self, '_neural_network'):
            raise ValueError(
                "_FeatureSelectionMixin can only be used with classes that have "
                "a '_neural_network' attribute."
            )

        self.lambda_qut_ = lambda_qut
        self.penalty = penalty

        # Initialize state
        self.penalized_parameters_ = None
        self.unpenalized_parameters_ = None
        self.selected_features_indices_ = None
        self.feature_names_in_ = None

    @property
    def selected_features_(self) -> List[str]:
        """Get names of selected features after training.

        Returns
        -------
        list of str
            List of selected feature names.
        """
        if self.feature_names_in_ is None:
            raise ValueError("Feature names not available.")

        if self.selected_features_indices_ is None:
            raise ValueError("Model not trained yet.")

        return [self.feature_names_in_[i] for i in self.selected_features_indices_]

    @property
    def n_features_in_(self) -> Optional[int]:
        """Get the number of input features.

        Returns
        -------
        int or None
            Number of input features, or None if not available.
        """
        return len(self.feature_names_in_) if self.feature_names_in_ is not None else None

    @property
    def coef_(self) -> np.ndarray:
        """Get the coefficients of the first layer.

        For models with hidden layers, returns the L2 norm of the weights
        connecting each input feature to the hidden layer.

        Returns
        -------
        ndarray
            Coefficient array of shape (n_features_in_,).
        """
        if not hasattr(self, '_neural_network') or self._neural_network.layers is None:
            raise ValueError("Model has not been initialized yet.")

        first_layer = self._neural_network.layers[0]
        weights = first_layer.weight.detach().cpu().numpy()

        # Handle weight tensor dimensions
        if weights.ndim == 2:
            if weights.shape[0] == 1:
                # Single output neuron
                weights = weights.squeeze(axis=0)
            elif weights.shape[1] == 1:
                # Single input feature
                weights = weights.squeeze(axis=1)
            else:
                # Multiple neurons - use L2 norm
                weights = np.linalg.norm(weights, axis=0)
                warnings.warn(
                    "Model has multiple hidden neurons. Coefficients computed "
                    "as L2 norm of weights from each input feature.",
                    UserWarning
                )
        elif weights.ndim == 1:
            pass
        else:
            raise ValueError(f"Unexpected weight tensor shape: {weights.shape}")

        if self.n_features_in_ is None:
            raise ValueError("Number of input features not available.")

        coefs = np.zeros(self.n_features_in_)

        if self.selected_features_indices_ is not None:
            for i, idx in enumerate(self.selected_features_indices_):
                coefs[idx] = weights[i]
        else:
            coefs = weights

        return coefs

    def _extract_feature_names(self, X: Union[np.ndarray, Any]) -> List[str]:
        """Extract feature names from input data.

        If no names are available, generates default names based on the
        number of features.

        Parameters
        ----------
        X : array-like
            Input data (numpy array, pandas DataFrame, etc.).

        Returns
        -------
        list of str
            List of feature names.
        """
        # Try pandas DataFrame first
        if hasattr(X, 'columns'):
            default_labels = list(range(X.shape[1]))
            cols = list(X.columns)
            if cols != default_labels:
                return cols

        # Try structured numpy array
        if isinstance(X, np.ndarray) and X.dtype.names is not None:
            return list(X.dtype.names)

        # Default to string indices
        return [f"feature_{i}" for i in range(X.shape[1])]

    def _compute_regularization_term(self, **reg_kwargs) -> torch.Tensor:
        """Compute regularization term based on penalty type.

        Parameters
        ----------
        **reg_kwargs
            Regularization parameters including lambda_, nu, a.

        Returns
        -------
        torch.Tensor
            Regularization loss tensor.
        """
        if self.penalized_parameters_ is None:
            return torch.tensor(0.0, requires_grad=True)

        device = next(self._neural_network.parameters()).device
        lambda_ = reg_kwargs.get('lambda_', self.lambda_qut_)

        if lambda_ == 0 or lambda_ is None:
            return torch.tensor(0.0, device=device)

        reg_loss = torch.tensor(0.0, device=device, requires_grad=True)

        for param in self.penalized_parameters_:
            abs_p = param.abs()

            if self.penalty == 'lasso':
                reg_loss = reg_loss + lambda_ * torch.sum(abs_p)

            elif self.penalty == 'harder':
                nu = reg_kwargs.get('nu', 0.1)
                eps = 1e-10
                abs_p_eps = abs_p + eps
                pow_term = torch.pow(abs_p_eps, 1 - nu)
                reg_loss = reg_loss + lambda_ * (abs_p_eps / (1 + pow_term)).sum()

            elif self.penalty == 'scad':
                a = reg_kwargs.get('a', 3.7)

                # SCAD penalty regions
                mask1 = abs_p <= lambda_
                mask2 = (abs_p > lambda_) & (abs_p <= a * lambda_)
                mask3 = abs_p > a * lambda_

                linear_part = (lambda_ * abs_p[mask1]).sum()

                numer = 2 * a * lambda_ * abs_p - param**2 - lambda_**2
                quadratic_part = (numer[mask2] / (2 * (a - 1))).sum()

                constant_part = (lambda_**2 * (a + 1) / 2) * mask3.sum()

                reg_loss = reg_loss + linear_part + quadratic_part + constant_part

        return reg_loss

    def _identify_parameter_types(self) -> None:
        """Identify which parameters should be penalized.

        Only the first layer weights are penalized for feature selection,
        while other weights and all biases are not penalized.
        """
        if not hasattr(self, '_neural_network') or self._neural_network.layers is None:
            raise ValueError("Neural network has not been initialized yet.")

        layers = self._neural_network.layers
        self.penalized_parameters_ = []
        self.unpenalized_parameters_ = []

        # All other parameters are not penalized
        for i, layer in enumerate(layers):
            if i == 0:
                self.penalized_parameters_.append(layer.weight)
            else:
                self.unpenalized_parameters_.append(layer.weight)

            # Add bias if present
            if layer.bias is not None:
                self.unpenalized_parameters_.append(layer.bias)

    def _count_nonzero_weights_first_layer(self) -> Tuple[int, List[int]]:
        """Count non-zero weights in the first layer.

        Returns
        -------
        tuple of (int, list)
            Count and indices of non-zero input connections.
        """
        if not hasattr(self, '_neural_network') or self._neural_network.layers is None:
            raise ValueError("Neural network has not been initialized yet.")

        first_layer = self._neural_network.layers[0]
        weight = first_layer.weight

        # Find non-zero columns (input features)
        non_zero_columns = weight.abs().sum(dim=0) > 0
        count = non_zero_columns.sum().item()
        indices = torch.nonzero(non_zero_columns, as_tuple=False).squeeze(1).tolist()

        return count, indices

    def _prune_features(self) -> None:
        """Prune network based on selected features and neurons.

        This method removes connections to input features that have zero weights
        and removes neurons that have no remaining connections.
        """
        if not hasattr(self, '_neural_network') or self._neural_network.layers is None:
            raise ValueError("Neural network has not been initialized yet.")

        layers = self._neural_network.layers
        _, active_features_idx = self._count_nonzero_weights_first_layer()

        first_layer = layers[0]
        pruned_weight = first_layer.weight.data[:, active_features_idx]

        # Find active neurons (those with non-zero weights)
        if self._neural_network.hidden_dims:
            active_neurons = torch.nonzero(
                ~torch.all(first_layer.weight == 0, dim=1),
                as_tuple=True
            )[0]
            pruned_weight = pruned_weight[active_neurons, :]
        else:
            active_neurons = None

        new_first_layer = torch.nn.Linear(
            in_features=pruned_weight.shape[1],
            out_features=pruned_weight.shape[0],
            bias=first_layer.bias is not None
        )
        new_first_layer.weight.data = pruned_weight
        if first_layer.bias is not None:
            new_first_layer.bias.data = first_layer.bias.data[active_neurons]

        layers[0] = new_first_layer

        # Prune second layer if it exists
        if len(layers) > 1 and active_neurons is not None:
            second_layer = layers[1]
            old_linear = second_layer.linear
            pruned_weight_2 = old_linear.weight.data[:, active_neurons]

            new_linear = torch.nn.Linear(
                in_features=pruned_weight_2.shape[1],
                out_features=pruned_weight_2.shape[0],
                bias=old_linear.bias is not None
            )
            new_linear.weight.data = pruned_weight_2
            if old_linear.bias is not None:
                new_linear.bias.data = old_linear.bias.data
            second_layer.linear = new_linear

        # Store selected features
        self.selected_features_indices_ = active_features_idx
        self._identify_parameter_types()

    def visualize_coefficients(
        self,
        figsize: Tuple[int, int] = (12, 8),
        save_path: Optional[str] = None,
        sorted_by_magnitude: bool = True
    ) -> None:
        """Visualize the non-zero coefficients of the model.

        Parameters
        ----------
        figsize : tuple of int, default=(12, 8)
            Figure size as (width, height).
        save_path : str, optional
            Path to save the figure. If None, displays the plot.
        sorted_by_magnitude : bool, default=True
            Whether to sort features by coefficient magnitude.
        """
        from .._utils._visuals import plot_lollipop

        if not hasattr(self, '_neural_network') or self._neural_network.layers is None:
            raise ValueError("Model not trained yet.")

        if self.selected_features_indices_ is None:
            raise ValueError("Model not trained yet.")

        # Get coefficients
        first_layer = self._neural_network.layers[0]
        if hasattr(first_layer, 'linear'):
            coefs = first_layer.linear.weight.detach().cpu().numpy()
        else:
            coefs = first_layer.weight.detach().cpu().numpy()

        # Handle multi-dimensional coefficients
        if coefs.shape[0] != 1:
            warnings.warn(
                "Model has multiple hidden neurons. Coefficients computed "
                "as L2 norm of weights from each input feature.",
                UserWarning
            )
            coefs = np.linalg.norm(coefs, axis=0)

        coefs = coefs.squeeze()

        # Get feature names
        feature_names = [
            self.feature_names_in_[idx]
            for idx in self.selected_features_indices_
        ]

        # Create plot
        plot_lollipop(
            coefficients=coefs,
            feature_names=feature_names,
            sorted_by_magnitude=sorted_by_magnitude,
            figsize=figsize,
            save_path=save_path
        )

    def get_feature_importance(self) -> Dict[str, float]:
        """Get feature importance scores.

        Returns
        -------
        dict
            Dictionary mapping feature names to importance scores.
        """
        if not hasattr(self, '_neural_network') or self._neural_network.layers is None:
            raise ValueError("Model not trained yet.")

        if self.selected_features_indices_ is None:
            raise ValueError("Model not trained yet.")

        # Get coefficients
        first_layer = self._neural_network.layers[0]
        if hasattr(first_layer, 'linear'):
            coefs = first_layer.linear.weight.detach().cpu().numpy()
        else:
            coefs = first_layer.weight.detach().cpu().numpy()

        # Handle multi-dimensional coefficients
        if coefs.shape[0] != 1:
            coefs = np.linalg.norm(coefs, axis=0)

        coefs = np.abs(coefs.squeeze())

        # Create importance dictionary
        importance = {}
        for i, idx in enumerate(self.selected_features_indices_):
            feature_name = self.feature_names_in_[idx]
            importance[feature_name] = float(coefs[i])

        return importance

    def get_selection_mask(self) -> np.ndarray:
        """Get boolean mask indicating selected features.

        Returns
        -------
        ndarray
            Boolean array where True indicates selected features.
        """
        if self.n_features_in_ is None:
            raise ValueError("Number of input features not available.")

        mask = np.zeros(self.n_features_in_, dtype=bool)

        if self.selected_features_indices_ is not None:
            mask[self.selected_features_indices_] = True

        return mask

    def transform(self, X: Union[np.ndarray, Any]) -> np.ndarray:
        """Transform input by selecting only the chosen features.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Input data.

        Returns
        -------
        ndarray
            Transformed data with only selected features.
        """
        if self.selected_features_indices_ is None:
            raise ValueError("Model not trained yet.")

        if hasattr(X, 'iloc'):
            # pandas DataFrame
            return X.iloc[:, self.selected_features_indices_].values
        else:
            # numpy array
            return X[:, self.selected_features_indices_]

    def fit_transform(self, X: Union[np.ndarray, Any], y: Union[np.ndarray, Any]) -> np.ndarray:
        """Fit the model and transform the input data.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Input features.
        y : array-like
            Target values.

        Returns
        -------
        ndarray
            Transformed input data with selected features.
        """
        self.fit(X, y)
        return self.transform(X)

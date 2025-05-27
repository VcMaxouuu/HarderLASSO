import torch
import torch.nn as nn
import numpy as np
import warnings
from .BasicNN import BasicNN

class FeatureSelectionMixin:
    """
    A mixin class for feature selection functionality in neural networks.

    This mixin is designed to be used with a class having a `NN` attribute.
    It provides methods for feature selection, regularization, and pruning of network weights.

    Attributes:
        lambda_qut (float): Regularization parameter for controlling sparsity.
        nu (float): Regularization parameter for controlling the penalty.
        penalized_parameters_ (list): List of parameters to penalize during training.
        unpenalized_parameters_ (list): List of parameters not penalized during training.
        selected_features_idx_ (list or None): Indices of selected features.
        input_features_ (list or None): Names of the given features.
        selected_features_ (list or None): Names of the selected features.
    """
    def __init__(self, lambda_qut=None, nu: float = 0.1):
        """
        Initializes the FeatureSelectionMixin class.

        Args:
            lambda_qut (float, optional): Regularization parameter for controlling sparsity. Defaults to None.
            nu (float, optional): Regularization parameter for controlling the penalty type. Defaults to 0.1.

        Raises:
            ValueError: If the class does not have a `NN` attribute.
        """
        if not hasattr(self, 'NN'):
            raise ValueError("FeatureSelectionMixin can only be used with a class that has a `NN` attribute.")

        self.lambda_qut = lambda_qut
        self.nu = nu

        self.penalized_parameters_    = None
        self.unpenalized_parameters_  = None
        self.selected_features_idx_   = None
        self.in_features_             = None

    @property
    def selected_features_(self):
        if self.in_features_ is None:
            raise ValueError("Feature names were not saved during fit.")
        if self.selected_features_idx_ is None:
            raise ValueError("Model not trained yet.")
        return [self.in_features_[i] for i in self.selected_features_idx_]

    @property
    def n_features_in_(self):
        if self.NN.layers is None:
            raise ValueError("`FeatureSelectionMixin` is used on a model that has not been initialized yet.")
        return len(self.in_features_) if self.in_features_ is not None else None


    @property
    def penalty_value_(self):
        return self._regularization_term(self.lambda_qut, self.nu).item()


    @property
    def coef_(self):
        """
        Returns the coefficients of the first layer of the neural network.
        If the model has a hidden layer, it returns the l2 norm of the weights.
        """
        if self.NN.layers is None:
            raise ValueError("`FeatureSelectionMixin` is used on a model that has not been initialized yet.")

        non_zero = self.NN.layers[0].weight.detach().cpu().numpy()
        if non_zero.shape[0] != 1:
            raise UserWarning(
                "Model is not a simple linear model, call `model.NN.layers[0].weight` to get the coefficients of first layer."
            )

        non_zero = non_zero.squeeze()
        coefs = np.zeros(shape=(self.n_features_in_,))
        for idx in self.selected_features_idx_:
            coefs[idx] = non_zero[idx]

        return coefs

    def _regularization_term(self, lambda_, nu):
        """Compute the regularization loss based on penalized parameters."""
        if lambda_ == 0:
            return torch.tensor(0.0, device=next(self.NN.parameters()).device)

        reg_loss = torch.tensor(0.0, device=next(self.NN.parameters()).device, requires_grad=True)
        for param in self.penalized_parameters_:
            if nu is None:
                reg_loss = reg_loss + torch.sum(torch.abs(param))
            else:
                abs_param = torch.abs(param)
                pow_term = torch.pow(abs_param + 1e-10, 1 - nu)
                reg_loss = reg_loss + torch.sum((abs_param + 1e-10) / (1 + pow_term))

        return lambda_ * reg_loss

    def _get_params_types(self) -> None:
        """Identify which parameters to penalize and which not."""
        if self.NN.layers is None:
            raise ValueError("`FeatureSelectionMixin` is used on a model that has not been initialized yet.")

        self.penalized_parameters_ = []
        self.unpenalized_parameters_ = []

        for i, layer in enumerate(self.NN.layers):
            if i == 0:
                self.penalized_parameters_.append(layer.weight)
            else:
                self.unpenalized_parameters_.append(layer.weight)

            if layer.bias is not None:
                self.unpenalized_parameters_.append(layer.bias)

    def _compute_regularization_paths(self, X_tensor, target_tensor, path_values=None):
        """
        Computes the lambda and nu regularization paths.

        Args:
            X_tensor (torch.Tensor): Input tensor.
            target_tensor (torch.Tensor): Target tensor.
            path_values (list[float], optional): Custom lambda path values. Defaults to None.

        Returns:
            tuple: Lambda path and nu path lists.

        Raises:
            ValueError: If QUT is not provided to compute the lambda path.
        """
        if not hasattr(self, 'QUT'):
            raise ValueError("QUT must be provided to compute the lambda path.")

        lambda_path = self.QUT.compute_lambda_path(
            X_tensor,
            target_tensor,
            path_values=path_values,
            hidden_dims=self.NN.hidden_dims,
            activation=self.NN.activation
        )

        self.lambda_qut = self.QUT.lambda_qut_

        if self.nu is None:
            nu_path = [None] * len(lambda_path)
        else:
            if len(lambda_path) == 6:
                # Default path with six values
                nu_path = [1, 0.7, 0.5, 0.3, 0.2, 0.1]
            else:
                nu_path = np.round(np.logspace(0, -1, num=len(lambda_path)), 1).tolist()

        return lambda_path, nu_path

    def _count_nonzero_weights_layer0(self) -> tuple:
        """Count non-zero weights in the first layer and return their indices."""
        if self.NN.layers is None:
            raise ValueError("`FeatureSelectionMixin` is used on a model that has not been initialized yet.")

        layer0_weight = self.NN.layers[0].weight
        non_zero_columns = layer0_weight.abs().sum(dim=0) > 0
        count = non_zero_columns.sum().item()
        indices = torch.nonzero(non_zero_columns, as_tuple=False).squeeze(1).tolist()
        return count, indices

    def _prune(self) -> None:
        """Prune network weights based on selected features and neurons."""
        NN = self.NN
        if NN.layers is None:
            raise ValueError("`FeatureSelectionMixin` is used on a model that has not been initialized yet.")

        layer0 = NN.layers[0]
        active_features_idx = self._count_nonzero_weights_layer0()[1]

        # Prune the first layer
        pruned_weight_0 = layer0.weight.data[:, active_features_idx]
        if NN.hidden_dims is not None:
            active_neurons = torch.nonzero(~torch.all(layer0.weight == 0, dim=1), as_tuple=True)[0]
            pruned_weight_0 = pruned_weight_0[active_neurons, :]
            layer0.out_features = pruned_weight_0.shape[0]
        else:
            active_neurons = None

        layer0.weight.data = pruned_weight_0
        layer0.in_features = pruned_weight_0.shape[1]

        if layer0.bias is not None:
            pruned_bias_0 = layer0.bias.data[active_neurons] if active_neurons is not None else layer0.bias.data
            layer0.bias.data = pruned_bias_0

        # Prune the second layer if hidden_dims is not None
        layer1 = NN.layers[1] if NN.hidden_dims is not None else None
        if layer1 is not None:
            layer1 = layer1.linear
            pruned_weight_1 = layer1.weight.data[:, active_neurons].clone()
            layer1.weight = nn.Parameter(pruned_weight_1)
            layer1.in_features = pruned_weight_1.shape[1]

        self.selected_features_idx_ = active_features_idx

    def _get_column_names(self, X):
        from pandas import DataFrame
        if isinstance(X, DataFrame):
            default_labels = list(range(X.shape[1]))
            cols = list(X.columns)
            return cols if cols != default_labels else None

        if isinstance(X, np.ndarray) and X.dtype.names is not None:
            return list(X.dtype.names)

        return np.arange(X.shape[1]).astype(str).tolist()

    def visualize_nonzero_coefficients(self, figsize=(12, 8), save_path=None):
        from utils.visuals.network_plots import plot_lollipop

        if self.NN.layers is None or self.selected_features_idx_ is None:
            raise ValueError("Model not trained yet.")

        coefs = self.NN.layers[0].weight.detach().cpu().numpy()

        if coefs.shape[0] != 1:
            warnings.warn(
                "Model has a hidden layer, the coefficients are based on l2 norm of the weights.",
                UserWarning
            )
            coefs = np.linalg.norm(coefs, axis=0)
        coefs = coefs.squeeze()
        features_names = [self.in_features_[idx] for idx in self.selected_features_idx_]
        plot_lollipop(coefs, features_names, sorted=False, figsize=figsize, save_path=save_path)

    def feature_importance(self, figsize=(12, 8), save_path=None):
        from utils.visuals.network_plots import plot_lollipop

        if self.NN.layers is None or self.selected_features_idx_ is None:
            raise ValueError("Model not trained yet.")

        coefs = self.NN.layers[0].weight.detach().cpu().numpy()

        if coefs.shape[0] != 1:
            warnings.warn(
                "Model has a hidden layer, the coefficients are based on l2 norm of the weights.",
                UserWarning
            )
            coefs = np.linalg.norm(coefs, axis=0)
        coefs = coefs.squeeze()
        features_names = [self.in_features_[idx] for idx in self.selected_features_idx_]
        plot_lollipop(coefs, features_names, sorted=True, figsize=figsize, save_path=save_path)

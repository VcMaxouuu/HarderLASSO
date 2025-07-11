"""
Neural network implementation for HarderLASSO models.

This module provides the _NeuralNetwork class which implements the core
neural network architecture used in HarderLASSO models.
"""

import torch
import torch.nn as nn
from typing import Optional, Tuple, List, Union


class _NeuralNetwork(nn.Module):
    """Neural network implementation for HarderLASSO models.

    This class implements a feedforward neural network with customizable
    architecture, supporting hidden layers with activation functions and
    proper weight initialization.

    Parameters
    ----------
    hidden_dims : tuple of int, optional
        Number of neurons in each hidden layer. If None or empty, creates
        a linear model.
    output_dim : int, optional
        Number of output features. If None, will be set when building layers.
    bias : bool, default=True
        Whether to include bias terms in layers.
    activation : nn.Module, optional
        Activation function to use between layers. If None, defaults to ReLU.

    Attributes
    ----------
    hidden_dims : tuple of int
        Architecture of hidden layers.
    output_dim : int
        Number of output features.
    bias : bool
        Whether bias terms are included.
    activation : nn.Module
        Activation function used between layers.
    layers : nn.ModuleList
        Network layers.

    Examples
    --------
    >>> network = _NeuralNetwork(hidden_dims=(10, 5), output_dim=1, bias=True)
    >>> network._build_layers(input_dim=20)
    >>> network._initialize_layers()
    >>> output = network(torch.randn(32, 20))
    """

    def __init__(
        self,
        hidden_dims: Optional[Tuple[int, ...]] = None,
        output_dim: Optional[int] = None,
        bias: bool = True,
        activation: Optional[nn.Module] = None
    ):
        """Initialize the neural network.

        Parameters
        ----------
        hidden_dims : tuple of int, optional
            Number of neurons in each hidden layer.
        output_dim : int, optional
            Number of output features.
        bias : bool, default=True
            Whether to include bias terms in layers.
        activation : nn.Module, optional
            Activation function between layers.
        """
        super().__init__()

        # Validate inputs
        if hidden_dims is not None and any(dim <= 0 for dim in hidden_dims):
            raise ValueError("All hidden dimensions must be positive integers.")

        if output_dim is not None and output_dim <= 0:
            raise ValueError("Output dimension must be a positive integer.")

        # Store configuration
        self._hidden_dims = hidden_dims or ()
        self._output_dim = output_dim
        self.bias = bias

        # Network components
        self.activation = activation or nn.ReLU() #TODO: check that activation is unbounded
        self.layers = None

        # State tracking
        self._is_built = False

    @property
    def input_dim(self) -> Optional[int]:
        """Get the input dimension of the network.

        Returns
        -------
        int or None
            Input dimension if layers are built, None otherwise.
        """
        if self.layers is not None and len(self.layers) > 0:
            return self.layers[0].in_features
        return None

    @property
    def output_dim(self) -> Optional[int]:
        """Get the output dimension of the network.

        Returns
        -------
        int or None
            Output dimension if layers are built, None otherwise.
        """
        if self.layers is not None and len(self.layers) > 0:
            return self.layers[-1].out_features
        return self._output_dim

    @property
    def hidden_dims(self) -> Optional[Tuple[int, ...]]:
        """Get the hidden layer dimensions.

        Returns
        -------
        tuple of int or None
            Hidden layer dimensions if available.
        """
        if self.layers is not None and len(self.layers) > 1:
            return tuple(layer.out_features for layer in self.layers[:-1])
        elif self.layers is not None and len(self.layers) == 1:
            return ()
        return self._hidden_dims

    @property
    def n_parameters(self) -> int:
        """Get the total number of parameters in the network.

        Returns
        -------
        int
            Total number of trainable parameters.
        """
        if self.layers is None:
            return 0
        return sum(p.numel() for p in self.parameters())

    def _build_layers(self, input_dim: int) -> None:
        """Build the neural network layers.

        Creates the layer architecture based on the specified dimensions.
        Uses normalized linear layers for hidden and output layers.

        Parameters
        ----------
        input_dim : int
            Number of input features.
        """
        if input_dim <= 0:
            raise ValueError("Input dimension must be positive.")

        if self._is_built:
            return

        from .._utils import _NormalizedLinearWrapper

        layers = nn.ModuleList()

        # Get configuration
        hidden_dims = self._hidden_dims
        output_dim = self._output_dim
        bias = self.bias

        # Build architecture
        if not hidden_dims:
            layers.append(nn.Linear(input_dim, output_dim, bias=bias))
        else:
            layers.append(nn.Linear(input_dim, hidden_dims[0], bias=True))

            for i in range(1, len(hidden_dims)):
                layer = _NormalizedLinearWrapper(
                    nn.Linear(hidden_dims[i - 1], hidden_dims[i], bias=True)
                )
                layers.append(layer)

            output_layer = _NormalizedLinearWrapper(
                nn.Linear(hidden_dims[-1], output_dim, bias=bias)
            )
            layers.append(output_layer)

        self.layers = layers
        self._is_built = True

    def _initialize_layers(self) -> None:
        """Initialize network weights and biases.

        Uses Xavier uniform initialization for weights and zeros for biases.
        """
        if not self._is_built:
            raise RuntimeError("Layers must be built before initialization.")

        for layer in self.layers:
            if hasattr(layer, 'weight') and layer.weight is not None:
                nn.init.xavier_uniform_(layer.weight)

            if hasattr(layer, 'bias') and layer.bias is not None:
                nn.init.zeros_(layer.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the network.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (batch_size, input_dim).

        Returns
        -------
        torch.Tensor
            Output tensor of shape (batch_size, output_dim).
        """
        if not self._is_built:
            raise RuntimeError("Network must be built before forward pass.")

        for i, layer in enumerate(self.layers):
            x = layer(x)

            if i < len(self.layers) - 1:
                x = self.activation(x)

        return x.squeeze()

    def summary(self) -> None:
        """Print a summary of the network architecture.

        Displays information about layers, parameters, and network structure.
        """
        print("=== Neural Network Summary ===")

        if not self._is_built:
            print("Network not built yet.")
            return

        print(f"Input dimension: {self.input_dim}")
        print(f"Output dimension: {self.output_dim}")
        print(f"Hidden dimensions: {self.hidden_dims}")
        print(f"Total parameters: {self.n_parameters:,}")
        print(f"Activation: {self.activation.__class__.__name__}")

        print("\nLayer Details:")
        for i, layer in enumerate(self.layers):
            print(f"  Layer {i}: {layer} "
                  f"({layer.in_features} -> {layer.out_features})")

    def reset_parameters(self) -> None:
        """Reset all network parameters to their initial values."""
        if self._is_built:
            self._initialize_layers()

    def __repr__(self) -> str:
        """String representation of the network."""
        if not self._is_built:
            return f"_NeuralNetwork(hidden_dims={self._hidden_dims}, output_dim={self._output_dim}, bias={self.bias})"
        return f"_NeuralNetwork(input_dim={self.input_dim}, hidden_dims={self.hidden_dims}, output_dim={self.output_dim})"

"""
Preprocessing utilities for HarderLASSO models.

This module contains preprocessing utilities including normalized linear layers
and data transformation functions.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class _NormalizedLinearWrapper(nn.Module):
    """
    Wrapper for linear layers that normalizes weights during forward pass.

    This wrapper normalizes the weights of a linear layer using L2 normalization
    during the forward pass, which can help with training stability and convergence.

    Parameters
    ----------
    linear : nn.Linear
        The linear layer to wrap.

    Attributes
    ----------
    linear : nn.Linear
        The underlying linear layer.

    Examples
    --------
    >>> linear = nn.Linear(10, 5)
    >>> normalized = _NormalizedLinearWrapper(linear)
    >>> output = normalized(torch.randn(32, 10))
    """

    def __init__(self, linear: nn.Linear):
        """Initialize the normalized linear wrapper."""
        super().__init__()

        if not isinstance(linear, nn.Linear):
            raise ValueError("Expected nn.Linear layer, got {}".format(type(linear)))

        self.linear = linear

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with normalized weights.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (batch_size, in_features).

        Returns
        -------
        torch.Tensor
            Output tensor of shape (batch_size, out_features).
        """
        # Normalize weights using L2 norm along the input dimension
        normalized_weight = F.normalize(self.linear.weight, p=2, dim=1)

        # Apply linear transformation with normalized weights
        return F.linear(x, normalized_weight, self.linear.bias)

    @property
    def weight(self) -> torch.Tensor:
        """Access the underlying weight tensor."""
        return self.linear.weight

    @property
    def bias(self) -> Optional[torch.Tensor]:
        """Access the underlying bias tensor."""
        return self.linear.bias

    @property
    def in_features(self) -> int:
        """Get the number of input features."""
        return self.linear.in_features

    @property
    def out_features(self) -> int:
        """Get the number of output features."""
        return self.linear.out_features

    @in_features.setter
    def in_features(self, value: int):
        """Set the number of input features."""
        self.linear.in_features = value

    @out_features.setter
    def out_features(self, value: int):
        """Set the number of output features."""
        self.linear.out_features = value

    def parameters(self):
        """Return parameters of the underlying linear layer."""
        return self.linear.parameters()

    def named_parameters(self, prefix: str = '', recurse: bool = True):
        """Return named parameters of the underlying linear layer."""
        return self.linear.named_parameters(prefix, recurse)

    def state_dict(self, destination=None, prefix='', keep_vars=False):
        """Return state dict of the underlying linear layer."""
        return self.linear.state_dict(destination, prefix, keep_vars)

    def load_state_dict(self, state_dict, strict=True):
        """Load state dict into the underlying linear layer."""
        return self.linear.load_state_dict(state_dict, strict)

    def extra_repr(self) -> str:
        """Extra representation for debugging."""
        return f'in_features={self.in_features}, out_features={self.out_features}, bias={self.bias is not None}'

    def __repr__(self) -> str:
        """String representation of the layer."""
        return f'_NormalizedLinearWrapper(in_features={self.in_features}, out_features={self.out_features}, bias={self.bias is not None})'

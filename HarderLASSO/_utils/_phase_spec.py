"""
Training phase specification for HarderLASSO models.

This module defines the _PhaseSpec class which specifies the configuration
for each phase of the multi-phase training process.
"""

from dataclasses import dataclass, field
from typing import Callable, Dict, Iterable, Any, Optional
import torch


@dataclass
class _PhaseSpec:
    """
    Specification for a single training phase.

    This class encapsulates all the configuration needed for one phase
    of the multi-phase HarderLASSO training process.

    Parameters
    ----------
    optimizer_class : callable
        Class of optimizer to use (e.g., torch.optim.Adam).
    parameter_getter : callable
        Function that returns parameters to optimize.
    optimizer_kwargs : dict, default={}
        Arguments passed to the optimizer constructor.
    regularization_kwargs : dict, default={}
        Regularization parameters for this phase.
    relative_tolerance : float, default=1e-5
        Convergence tolerance for relative loss change.
    max_epochs : int, optional
        Maximum number of epochs for this phase.

    Examples
    --------
    >>> phase = _PhaseSpec(
    ...     optimizer_class=torch.optim.Adam,
    ...     parameter_getter=lambda: model.parameters(),
    ...     optimizer_kwargs={'lr': 0.01},
    ...     regularization_kwargs={'lambda_': 0.1},
    ...     relative_tolerance=1e-5,
    ...     max_epochs=1000
    ... )
    """

    # Optimizer class to instantiate
    optimizer_class: Callable[..., torch.optim.Optimizer]

    # Function that returns parameters to optimize
    parameter_getter: Callable[[], Iterable[torch.nn.Parameter]]

    # Arguments passed to the optimizer constructor
    optimizer_kwargs: Dict[str, Any] = field(default_factory=dict)

    # Regularization hyperparameters for this phase
    regularization_kwargs: Dict[str, Any] = field(default_factory=dict)

    # Convergence tolerance for relative loss change
    relative_tolerance: float = 1e-5

    # Maximum number of epochs (None for unlimited)
    max_epochs: Optional[int] = None

    def __post_init__(self):
        """Validate the phase specification after initialization."""
        if self.relative_tolerance <= 0:
            raise ValueError("Relative tolerance must be positive.")

        if self.max_epochs is not None and self.max_epochs <= 0:
            raise ValueError("Maximum epochs must be positive if specified.")

        if not callable(self.optimizer_class):
            raise ValueError("Optimizer class must be callable.")

        if not callable(self.parameter_getter):
            raise ValueError("Parameter getter must be callable.")

    def __str__(self) -> str:
        """String representation of the phase specification."""
        return (
            f"_PhaseSpec("
            f"optimizer={self.optimizer_class.__name__}, "
            f"params={len(list(self.parameter_getter()))}, "
            f"reg_params={list(self.regularization_kwargs.keys())}, "
            f"tol={self.relative_tolerance}, "
            f"max_epochs={self.max_epochs})"
        )

    def __repr__(self) -> str:
        """Detailed representation of the phase specification."""
        return self.__str__()

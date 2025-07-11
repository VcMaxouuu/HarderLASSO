"""Base task mixin for neural network feature selection.

This module provides the abstract base class for task-specific mixins
in the HarderLASSO framework.
"""

from abc import ABC, abstractmethod
from typing import Dict, Tuple, Union, Optional, Any
import numpy as np
import torch


class _BaseTaskMixin(ABC):
    """Abstract base class for task-specific functionality.

    This mixin defines the interface that all task-specific implementations
    must follow. It provides the contract for preprocessing data, defining
    loss criteria, scoring predictions, and formatting outputs.

    Notes
    -----
    This is an internal class and should not be used directly by end users.
    """

    @abstractmethod
    def preprocess_data(
        self,
        X: np.ndarray,
        target: Optional[np.ndarray] = None,
        fit_scaler: bool = True
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """Preprocess input data and target for the specific task.

        This method should normalize input features and convert them to PyTorch
        tensors. Task-specific preprocessing should be implemented in subclasses.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Input features.
        target : ndarray, optional
            Target values. Shape depends on the specific task.
        fit_scaler : bool, default=True
            Whether to fit a scaler for normalization.

        Returns
        -------
        torch.Tensor or tuple of torch.Tensor
            If target is None, returns preprocessed features.
            If target is provided, returns tuple of (features, target).
        """
        ...

    @abstractmethod
    def criterion(self, outputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Define the loss function for the task.

        Parameters
        ----------
        outputs : torch.Tensor
            Model predictions. Shape depends on the specific task.
        targets : torch.Tensor
            Ground truth target values. Shape depends on the specific task.

        Returns
        -------
        torch.Tensor
            Computed loss value as scalar tensor.
        """
        ...

    @abstractmethod
    def score(self, X: np.ndarray, target: np.ndarray) -> Dict[str, float]:
        """Compute evaluation metrics for the task.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Input features.
        target : ndarray
            Ground truth target values. Shape depends on the specific task.

        Returns
        -------
        dict
            Dictionary of evaluation metrics and their values.
        """
        ...

    @abstractmethod
    def format_predictions(self, raw_outputs: np.ndarray) -> np.ndarray:
        """Format raw model outputs into user-friendly predictions.

        This method converts raw model outputs into the final prediction
        format expected by users.

        Parameters
        ----------
        raw_outputs : ndarray
            Raw outputs from the model. Shape depends on the task.

        Returns
        -------
        ndarray
            Formatted predictions in the expected format for the task.
        """
        ...

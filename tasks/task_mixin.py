from abc import ABC, abstractmethod
import numpy as np
import torch

class TaskMixin(ABC):
    """
    A mixin class defining the interface for task-specific functionality.

    Classes inheriting from this mixin must implement the abstract methods
    for preprocessing data, defining a loss criterion, scoring, and formatting predictions.
    """

    @abstractmethod
    def preprocess_data(self,
                   X: np.ndarray,
                   target: np.ndarray | None = None,
                   fit_scaler: bool = True) -> tuple[torch.Tensor, torch.Tensor | None]:
        """
        Preprocesses the input data and target for use in the model.

        Normalizes the input features using a `StandardScaler` and converts them to PyTorch tensors.
        The target is also converted to a PyTorch tensor if provided.


        Args:
            X (np.ndarray): The input features as a NumPy array.
            target (np.ndarray, optional): The target values as a NumPy array. Defaults to None.
            fit_scaler (bool, optional): Whether to fit a scaler for normalization. Defaults to True.

        Returns:
            tuple[torch.Tensor, torch.Tensor | None]: A tuple containing the preprocessed input
            features as a PyTorch tensor and the target tensor (or None if no target is provided).
        """
        ...

    @abstractmethod
    def criterion(self,
                outputs: torch.Tensor,
                targets: torch.Tensor) -> torch.Tensor:
        """
        Defines the loss function for the task.

        Args:
            outputs (torch.Tensor): The model's predictions.
            targets (torch.Tensor): The ground truth target values.

        Returns:
            torch.Tensor: The computed loss value.
        """
        ...

    @abstractmethod
    def score(self,
            X: np.ndarray,
            target: np.ndarray) -> dict[str, float]:
        """
        Computes evaluation metrics for the task.

        Args:
            X (np.ndarray): The input features as a NumPy array.
            target (np.ndarray): The ground truth target values as a NumPy array.

        Returns:
            dict[str, float]: A dictionary of evaluation metrics and their values.
        """
        ...

    @abstractmethod
    def format_predictions(self,
                    raw_outputs: np.ndarray) -> np.ndarray:
        """
        Formats raw model outputs into a user-friendly format specific to the task.

        Args:
            raw_outputs (np.ndarray): The raw outputs from the model.

        Returns:
            np.ndarray: The formatted predictions.
        """
        ...

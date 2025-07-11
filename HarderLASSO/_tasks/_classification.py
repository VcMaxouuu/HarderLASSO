"""Classification task mixin for neural network feature selection.

This module provides the classification-specific implementation of the task mixin
interface for the HarderLASSO framework.
"""

import torch
from numpy import ndarray, asarray, float32, argmax, unique
from sklearn.preprocessing import StandardScaler
from typing import Dict, Tuple, Optional, Union

from ._base import _BaseTaskMixin
from .._utils import _ClassificationLoss


class _ClassificationTaskMixin(_BaseTaskMixin):
    """Mixin class for classification tasks.

    This class provides implementations for preprocessing data, defining
    a classification-specific loss criterion, scoring predictions, and
    formatting outputs.

    Attributes
    ----------
    scaler : StandardScaler
        Feature preprocessing scaler.
    _loss : _ClassificationLoss
        Classification loss function.
    """

    def __init__(self) -> None:
        """Initialize the classification task mixin.

        Sets up the scaler for preprocessing input features and the loss function.
        """
        super().__init__()
        self.scaler: Optional[StandardScaler] = None
        self._loss = _ClassificationLoss(reduction='sum')

    def preprocess_data(
        self,
        X: ndarray,
        target: Optional[ndarray] = None,
        fit_scaler: bool = True
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """Preprocess input data for classification.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Input features.
        target : ndarray of shape (n_samples,) or (n_samples, 1), optional
            Target labels.
        fit_scaler : bool, default=True
            Whether to fit the scaler on X.

        Returns
        -------
        torch.Tensor or tuple of torch.Tensor
            If target is None, returns preprocessed features.
            If target is provided, returns tuple of (features, labels).
        """
        if self.scaler is None and fit_scaler:
            self.scaler = StandardScaler()

        X = asarray(X, dtype=float32)
        if X.ndim == 1:
            X = X.reshape(-1, 1)

        if fit_scaler:
            X_tensor = torch.tensor(self.scaler.fit_transform(X), dtype=torch.float)
        else:
            X_tensor = torch.tensor(self.scaler.transform(X), dtype=torch.float)

        if target is None:
            return X_tensor

        target = asarray(target, dtype=float32)
        if target.ndim == 2:
            if target.shape[1] == 1:
                target = target.squeeze(axis=1)
            else:
                raise ValueError(
                    f"Target should be of shape (n,) or (n, 1). Got: {target.shape}"
                )

        if hasattr(self._neural_network, '_output_dim'):
            self._neural_network._output_dim = len(unique(target))

        target_tensor = torch.tensor(target, dtype=torch.long)
        return X_tensor, target_tensor

    def criterion(self, outputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Compute the classification loss.

        Parameters
        ----------
        outputs : torch.Tensor of shape (n_samples, n_classes)
            Model predictions.
        targets : torch.Tensor of shape (n_samples,)
            True labels.

        Returns
        -------
        torch.Tensor
            Classification loss as scalar tensor.
        """
        return self._loss(outputs, targets)

    def score(self, X: ndarray, target: ndarray) -> Dict[str, float]:
        """Compute classification accuracy.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Input features.
        target : ndarray of shape (n_samples,) or (n_samples, 1)
            True labels.

        Returns
        -------
        dict
            Dictionary containing accuracy score.
        """
        target = asarray(target, dtype=float32)

        if target.ndim == 2:
            if target.shape[1] == 1:
                target = target.reshape(-1)
            else:
                raise ValueError(
                    f"Target should be of shape (n,) or (n, 1). Got: {target.shape}"
                )

        pred = self.predict(X)
        accuracy = (pred == target).mean()
        return {"accuracy": accuracy}

    def format_predictions(self, raw_outputs: ndarray) -> ndarray:
        """Format raw model outputs to class predictions.

        Parameters
        ----------
        raw_outputs : ndarray of shape (n_samples, n_classes)
            Raw model outputs (logits).

        Returns
        -------
        ndarray of shape (n_samples,)
            Class predictions.
        """
        outputs = raw_outputs.squeeze().cpu().numpy()
        return argmax(outputs, axis=1)

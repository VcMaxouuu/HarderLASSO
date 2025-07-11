"""Regression task mixin for neural network feature selection.

This module provides the regression-specific implementation of the task mixin
interface for the HarderLASSO framework.
"""

import torch
from numpy import ndarray, asarray, float32, argmax, unique
from sklearn.preprocessing import StandardScaler
from typing import Dict, Tuple, Optional, Union

from ._base import _BaseTaskMixin
from .._utils import _RSELoss


class _RegressionTaskMixin(_BaseTaskMixin):
    """Mixin class for regression tasks.

    This class provides implementations for preprocessing data, defining
    a regression-specific loss criterion, scoring predictions, and
    formatting outputs.

    Attributes
    ----------
    scaler : StandardScaler
        Feature preprocessing scaler.
    _loss : _RSELoss
        Regression loss function.
    """

    def __init__(self) -> None:
        """Initialize the regression task mixin.

        Sets up the scaler for preprocessing input features and the loss function.
        """
        super().__init__()
        self.scaler: Optional[StandardScaler] = None
        self._loss = _RSELoss(reduction='sum')

    def preprocess_data(
        self,
        X: ndarray,
        target: Optional[ndarray] = None,
        fit_scaler: bool = True
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """Preprocess input data for regression.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Input features.
        target : ndarray of shape (n_samples,) or (n_samples, 1), optional
            Regression target values.
        fit_scaler : bool, default=True
            Whether to fit the scaler on X.

        Returns
        -------
        torch.Tensor or tuple of torch.Tensor
            If target is None, returns preprocessed features.
            If target is provided, returns tuple of (features, target values).
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

        target_tensor = torch.tensor(target, dtype=torch.float)
        return X_tensor, target_tensor

    def criterion(self, outputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Compute the regression loss.

        Parameters
        ----------
        outputs : torch.Tensor of shape (n_samples,)
            Model predictions.
        targets : torch.Tensor of shape (n_samples,)
            Regression target values.

        Returns
        -------
        torch.Tensor
            Regression loss as scalar tensor.
        """
        return self._loss(outputs, targets)

    def score(self, X: ndarray, target: ndarray) -> Dict[str, float]:
        """Compute regression metrics.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Input features.
        target : ndarray of shape (n_samples,) or (n_samples, 1)
            Regression target values.

        Returns
        -------
        dict
            Dictionary containing MSE, MAE, and R² metrics.
        """
        from numpy import mean, sum, abs
        target = asarray(target, dtype=float32)

        if target.ndim == 2:
            if target.shape[1] == 1:
                target = target.reshape(-1)
            else:
                raise ValueError("Target should be of shape (n, ) or (n, 1). Got: {}".format(target.shape))

        pred = self.predict(X)
        mse = mean((target - pred) ** 2)
        mae = mean(abs(target - pred))
        target_variance = sum((target - mean(target)) ** 2)
        if target_variance == 0:
            r2 = float('nan')
        else:
            r2 = 1 - (sum((target - pred) ** 2) / target_variance)

        return {
            "MSE": mse,
            "MAE": mae,
            "R2": r2
        }

    def format_predictions(self, raw_outputs: ndarray) -> ndarray:
        """Format raw model outputs to regression predictions.

        Parameters
        ----------
        raw_outputs : ndarray of shape (n_samples, 1)
            Raw model outputs.

        Returns
        -------
        ndarray of shape (n_samples,)
            Regression predictions.
        """
        return  raw_outputs.squeeze().cpu().numpy()

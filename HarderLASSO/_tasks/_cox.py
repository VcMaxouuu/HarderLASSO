"""Cox regression task mixin for neural network feature selection.

This module provides the Cox regression-specific implementation of the task mixin
interface for the HarderLASSO framework.
"""

import torch
from numpy import ndarray, asarray, float32
from sklearn.preprocessing import StandardScaler
from typing import Dict, Tuple, Optional, Union

from ._base import _BaseTaskMixin
from .._utils import _CoxSquareLoss


class _CoxTaskMixin(_BaseTaskMixin):
    """Mixin class for Cox regression tasks.

    This class provides implementations for preprocessing survival data,
    defining a Cox-specific loss criterion, scoring predictions using
    concordance index, and formatting outputs.

    Parameters
    ----------
    approximation : {'Breslow', 'Efron'}, default='Breslow'
        Method for handling ties in Cox regression.

    Attributes
    ----------
    scaler : StandardScaler
        Feature preprocessing scaler.
    approximation : str
        Method for handling ties.
    _loss : _CoxSquareLoss
        Cox regression loss function.
    """

    def __init__(self, approximation: str = 'Breslow') -> None:
        """Initialize the Cox regression task mixin.

        Parameters
        ----------
        approximation : {'Breslow', 'Efron'}, default='Breslow'
            Method for handling ties in Cox regression.
        """
        super().__init__()
        self.scaler: Optional[StandardScaler] = None
        self.approximation = approximation
        self._loss = _CoxSquareLoss(reduction='sum', approximation=approximation)

    def preprocess_data(
        self,
        X: ndarray,
        target: Optional[ndarray] = None,
        fit_scaler: bool = True
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """Preprocess input data for Cox regression.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Input features.
        target : ndarray of shape (n_samples, 2), optional
            Survival data where first column is survival times and second
            column is event indicators.
        fit_scaler : bool, default=True
            Whether to fit the scaler on X.

        Returns
        -------
        torch.Tensor or tuple of torch.Tensor
            If target is None, returns preprocessed features.
            If target is provided, returns tuple of (features, survival data).
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
        if target.ndim != 2:
            raise ValueError(
                "Target should be a 2D array with columns (durations, events). "
                f"Got shape: {target.shape}"
            )

        # Handle transpose if needed (2 x n_samples -> n_samples x 2)
        if target.shape[0] == 2 and target.shape[1] != 2:
            target = target.T

        if target.shape[1] != 2:
            raise ValueError(
                "Target must have exactly two columns: (durations, events). "
                f"Got shape: {target.shape}"
            )

        target_tensor = torch.from_numpy(target)
        return X_tensor, target_tensor

    def criterion(self, outputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Compute the Cox regression loss.

        Parameters
        ----------
        outputs : torch.Tensor of shape (n_samples, 1)
            Model predictions (risk scores).
        targets : torch.Tensor of shape (n_samples, 2)
            Survival data where first column is survival times and second
            column is event indicators.

        Returns
        -------
        torch.Tensor
            Cox loss as scalar tensor.
        """
        return self._loss(outputs, targets)

    def score(self, X: ndarray, target: ndarray) -> Dict[str, float]:
        """Compute Cox regression evaluation metrics.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Input features.
        target : ndarray of shape (n_samples, 2)
            Survival data where first column is survival times and second
            column is event indicators.

        Returns
        -------
        dict
            Dictionary containing concordance index and negative partial
            log-likelihood.
        """
        # Import here to avoid circular imports
        from .._utils import concordance_index, cox_partial_log_likelihood

        target = asarray(target, dtype=float32)

        if target.ndim != 2:
            raise ValueError(
                "Target should be a 2D array with columns (durations, events). "
                f"Got shape: {target.shape}"
            )

        # Handle transpose if needed
        if target.shape[0] == 2 and target.shape[1] != 2:
            target = target.T

        times = target[:, 0]
        events = target[:, 1].astype(int)
        pred = self.predict(X)

        c_index = concordance_index(times, events, pred)
        npll = cox_partial_log_likelihood(times, events, pred)

        return {
            "C-index": c_index,
            "neg_partial_log_likelihood": npll
        }

    def format_predictions(self, raw_outputs: ndarray) -> ndarray:
        """Format raw model outputs to risk scores.

        Parameters
        ----------
        raw_outputs : ndarray of shape (n_samples, 1)
            Raw model outputs.

        Returns
        -------
        ndarray of shape (n_samples,)
            Risk scores.
        """
        return raw_outputs.squeeze().cpu().numpy()

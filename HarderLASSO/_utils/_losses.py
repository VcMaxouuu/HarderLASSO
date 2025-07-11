"""
Loss functions for HarderLASSO models.

This module contains specialized loss functions used in different HarderLASSO
model types for various tasks.
"""

import torch
import torch.nn as nn
import math
from typing import Optional, Literal


class _BaseLoss(nn.Module):
    """Base class for HarderLASSO loss functions."""

    def __init__(self, reduction: str = 'mean'):
        """Initialize base loss function."""
        super().__init__()

        if reduction not in ['mean', 'sum', 'none']:
            raise ValueError(f"Invalid reduction: {reduction}. Must be 'mean', 'sum', or 'none'.")

        self.reduction = reduction


class _RSELoss(_BaseLoss):
    """
    Root Square Error loss for regression tasks.

    This loss function computes the square root of the mean squared error,
    which can be more robust to outliers than standard MSE.

    Parameters
    ----------
    reduction : str, default='mean'
        Reduction method ('mean', 'sum', or 'none').

    Examples
    --------
    >>> loss_fn = _RSELoss()
    >>> predictions = torch.randn(32, 1)
    >>> targets = torch.randn(32)
    >>> loss = loss_fn(predictions, targets)
    """

    def __init__(self, reduction: str = 'mean'):
        """Initialize RSE loss function."""
        super().__init__(reduction)
        self.mse_loss = nn.MSELoss(reduction=reduction)

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Compute the root square error loss.

        Parameters
        ----------
        input : torch.Tensor
            Predicted values of shape (batch_size,) or (batch_size, 1).
        target : torch.Tensor
            True values of shape (batch_size,).

        Returns
        -------
        torch.Tensor
            Root square error loss.
        """
        mse = self.mse_loss(input, target)
        return torch.sqrt(mse)


class _ClassificationLoss(_BaseLoss):
    """
    Cross-entropy loss for classification tasks.

    This is a wrapper around PyTorch's CrossEntropyLoss with consistent
    interface for HarderLASSO models.

    Parameters
    ----------
    reduction : str, default='mean'
        Reduction method ('mean', 'sum', or 'none').

    Examples
    --------
    >>> loss_fn = _ClassificationLoss()
    >>> predictions = torch.randn(32, 10)  # 10 classes
    >>> targets = torch.randint(0, 10, (32,))
    >>> loss = loss_fn(predictions, targets)
    """

    def __init__(self, reduction: str = 'mean'):
        """Initialize classification loss function."""
        super().__init__(reduction)
        self.cross_entropy = nn.CrossEntropyLoss(reduction=reduction)

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Compute the cross-entropy loss.

        Parameters
        ----------
        input : torch.Tensor
            Predicted logits of shape (batch_size, n_classes).
        target : torch.Tensor
            True class indices of shape (batch_size,).

        Returns
        -------
        torch.Tensor
            Cross-entropy loss.
        """
        return self.cross_entropy(input, target)


class _CoxSquareLoss(_BaseLoss):
    """
    Cox proportional hazards loss for survival analysis.

    This loss function implements the Cox partial likelihood with support
    for tied event times using Breslow or Efron approximation.

    Parameters
    ----------
    reduction : str, default='mean'
        Reduction method ('mean', 'sum', or 'none').
    approximation : {'Breslow', 'Efron'}, default='Breslow'
        Method for handling tied event times.

    Examples
    --------
    >>> loss_fn = _CoxSquareLoss()
    >>> predictions = torch.randn(32)  # Risk scores
    >>> targets = torch.randn(32, 2)  # [times, events]
    >>> loss = loss_fn(predictions, targets)
    """

    def __init__(
        self,
        reduction: str = 'mean',
        approximation: Literal['Breslow', 'Efron'] = 'Breslow'
    ):
        """Initialize Cox loss function."""
        super().__init__(reduction)

        if approximation.lower() not in ['breslow', 'efron', None]:
            raise ValueError(f"Invalid approximation: {approximation}. Must be 'Breslow' or 'Efron' or None.")

        self.approximation = approximation
        self._initialized = False

    def _initialize(self, target: torch.Tensor) -> None:
        """Initialize loss function with target data structure."""
        times = target[:, 0]
        sort_idx = torch.argsort(-times)
        self.register_buffer('sort_idx', sort_idx)

        times = times[sort_idx]
        events = target[:, 1][sort_idx]
        device = times.device

        # Group by unique event times
        unique_times, inv_idx, counts = torch.unique_consecutive(
            times, return_inverse=True, return_counts=True
        )
        starts = torch.cat([counts.new_zeros(1), counts.cumsum(0)[:-1]])
        ends = counts.cumsum(0)

        # Count uncensored events per unique time
        sum_uncens = torch.zeros_like(unique_times, dtype=torch.float, device=device)
        for i in range(len(unique_times)):
            s, e = starts[i].item(), ends[i].item()
            sum_uncens[i] = events[s:e].sum()

        # Register buffers for state persistence
        self.register_buffer('events', events)
        self.register_buffer('starts', starts)
        self.register_buffer('ends', ends)
        self.register_buffer('counts', counts)
        self.register_buffer('unique_times', unique_times)
        self.register_buffer('sum_uncens', sum_uncens)

        self._initialized = True

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Compute the Square root Cox partial likelihood loss.

        Parameters
        ----------
        input : torch.Tensor
            Predicted risk scores of shape (batch_size,).
        target : torch.Tensor
            Target data of shape (batch_size, 2) with [times, events].

        Returns
        -------
        torch.Tensor
            Cox partial likelihood loss.
        """
        if input.dim() == 2 and input.shape[1] == 1:
            input = input.squeeze(1)

        if not self._initialized:
            self._initialize(target)

        input = input[self.sort_idx]

        # Case 1: No tied events or no approximation
        if self.approximation is None or (self.sum_uncens <= 1).all():
            log_cum = torch.logcumsumexp(input, dim=0)
            loss = - ((input - log_cum) * self.events).sum()

        # Case 2 : Tied events -> Using approximation
        if self.approximation == 'Breslow':
            scaled_log_cumsums = torch.logcumsumexp(input, dim=0)[self.ends - 1] * self.sum_uncens
            loss = - (input * self.events).sum() + scaled_log_cumsums.sum()

        if self.approximation == 'Efron':
            log_cum = torch.logcumsumexp(input, dim=0)
            single_idx = (self.sum_uncens == 1).nonzero().squeeze(1)
            loss_term = log_cum[self.ends - 1][single_idx].sum()

            # Terms for ties
            for idx in self.multi_event_idx.tolist():
                s, e = self.starts[idx].item(), self.ends[idx].item()
                d_i = int(self.sum_uncens[idx].item())
                # risk set is [0:e]
                local_max = input[:e].max()
                risk_sum = torch.exp(input[:e] - local_max).sum()
                tied_exp = torch.exp(input[s:e][self.events[s:e].bool()] - local_max).sum()
                denom = getattr(self, f'tied_denom_{idx}')
                for k in range(d_i):
                    efron = risk_sum - denom[k] * tied_exp
                    loss_term += torch.log(efron) + local_max

            loss = - (input * self.events).sum() + loss_term

        # Return negative log likelihood (to minimize)
        if self.reduction == 'mean':
            return torch.sqrt(loss / len(input))
        elif self.reduction == 'sum':
            return torch.sqrt(loss)


class _ExpGumbelLoss(_BaseLoss):
    """
    Gumbel distribution loss for extreme value modeling.

    This loss function implements the negative log-likelihood of the
    Gumbel distribution for modeling extreme values.

    Parameters
    ----------
    reduction : str, default='mean'
        Reduction method ('mean', 'sum', or 'none').

    Examples
    --------
    >>> loss_fn = _ExpGumbelLoss()
    >>> predictions = torch.randn(32, 2)  # [location, scale]
    >>> targets = torch.randn(32)
    >>> loss = loss_fn(predictions, targets)
    """

    def __init__(self, reduction: str = 'mean'):
        """Initialize Gumbel loss function."""
        super().__init__(reduction)
        self._beta_prev = None

    @staticmethod
    def _newton_beta(
        residual: torch.Tensor,
        beta0: torch.Tensor,
        max_iter: int = 20,
        tol: float = 1e-5) -> torch.Tensor:

        beta = beta0
        for _ in range(max_iter):
            s = residual/beta
            exp_neg_s = torch.exp(-s)

            f = beta - torch.mean(residual * (1 - exp_neg_s))
            df = 1 + torch.mean((residual**2) * exp_neg_s) / beta**2

            beta_new = beta - f / df
            if abs(beta_new - beta) < tol:
                beta = beta_new
                break

            beta = beta_new

        return beta

    def _estimate_beta(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            residual = target - input

            beta0 = residual.std() * (6**0.5) / torch.pi
            beta_hat = self._newton_beta(residual, beta0)
            self._beta_prev = beta_hat.detach()
            return beta_hat

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Compute the Exponential Gumbel distribution loss.

        Parameters
        ----------
        input : torch.Tensor
            Predicted parameters of shape (batch_size, 2) with [location, log_scale].
        target : torch.Tensor
            True values of shape (batch_size,).

        Returns
        -------
        torch.Tensor
            Gumbel distribution loss.
        """
        input = input.squeeze()
        beta = self._estimate_beta(input, target)
        s = (target - input) / beta
        loss = beta * (s.mean() + torch.exp(-s).mean()).exp()
        return loss

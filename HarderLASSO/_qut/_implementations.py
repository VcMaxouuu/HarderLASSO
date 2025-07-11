"""QUT implementations for specific tasks.

This module contains the concrete implementations of the QUT (Quantile Universal
Threshold) method for different machine learning tasks.
"""

import torch
from ._base import _BaseQUTMixin


class _RegressionQUT(_BaseQUTMixin):
    """
    QUT manager for regression tasks.

    This class implements the QUT method for regression problems, where the
    simulation is based on Gaussian noise around the mean.

    Parameters
    ----------
    lambda_qut : float, optional
        Custom lambda value for QUT. If None, will be computed automatically
        based on the quantile threshold.
    n_samples : int, default=5000
        Number of samples to use for QUT simulation.
    alpha : float, default=0.05
        Significance level for the quantile threshold.
    """

    def __init__(
        self,
        lambda_qut: float = None,
        n_samples: int = 5000,
        alpha: float = 0.05
    ) -> None:
        """Initialize the regression QUT manager."""
        super().__init__(lambda_qut, n_samples, alpha)

    def _simulation_method(
        self,
        X: torch.Tensor,
        target: torch.Tensor,
        n_reps: int
    ) -> torch.Tensor:
        """Simulate QUT values for regression.

        Args:
            X: Input features of shape (n_samples, n_features)
            target: Target values of shape (n_samples,)
            n_reps: Number of repetitions for simulation

        Returns:
            Simulated QUT values of shape (n_reps,)
        """
        device, dtype = X.device, X.dtype
        n_samples, _ = X.shape

        # Generate centered Gaussian noise
        y_sample = torch.randn(n_samples, n_reps, dtype=dtype, device=device)
        y_centered = y_sample - y_sample.mean(dim=0, keepdim=True)

        # Compute X^T * y for each repetition
        Xy = torch.matmul(X.T, y_centered)
        Xy_inf_norm = torch.abs(Xy).amax(dim=0)
        y_norm = y_centered.norm(p=2, dim=0)

        return Xy_inf_norm / y_norm


class _ClassificationQUT(_BaseQUTMixin):
    """QUT manager for classification tasks.

    This class implements the QUT method for classification problems, where the
    simulation is based on multinomial sampling from the empirical class distribution.

    Args:
        lambda_qut: Custom lambda value for QUT. If None, will be computed
            automatically based on the quantile threshold.
        n_samples: Number of samples to use for QUT simulation.
        alpha: Significance level for the quantile threshold.
    """

    def __init__(
        self,
        lambda_qut: float = None,
        n_samples: int = 5000,
        alpha: float = 0.05
    ) -> None:
        """Initialize the classification QUT manager."""
        super().__init__(lambda_qut, n_samples, alpha)

    def _compute_class_distribution(self, target: torch.Tensor) -> torch.Tensor:
        """Compute empirical class distribution from target labels.

        Args:
            target: Target labels of shape (n_samples,)

        Returns:
            Class probabilities of shape (n_classes,)
        """
        num_classes = target.unique().numel()
        counts = torch.bincount(target, minlength=num_classes).float()
        return counts / target.size(0)

    def _simulation_method(
        self,
        X: torch.Tensor,
        target: torch.Tensor,
        n_reps: int
    ) -> torch.Tensor:
        """Simulate QUT values for classification.

        Args:
            X: Input features of shape (n_samples, n_features)
            target: Target labels of shape (n_samples,)
            n_reps: Number of repetitions for simulation

        Returns:
            Simulated QUT values of shape (n_reps,)
        """
        device, dtype = X.device, X.dtype
        hat_p = self._compute_class_distribution(target).to(device=device, dtype=dtype)

        # Sample from multinomial distribution
        y_sample_indices = torch.multinomial(
            hat_p, X.shape[0] * n_reps, replacement=True
        )
        y_sample_indices = y_sample_indices.reshape(n_reps, X.shape[0])

        # Convert to one-hot encoding
        num_classes = len(hat_p)
        y_sample_one_hot = torch.nn.functional.one_hot(
            y_sample_indices, num_classes=num_classes
        ).float()

        # Center the samples
        y_centered = y_sample_one_hot - y_sample_one_hot.mean(dim=1, keepdim=True)

        # Compute X^T * y for each repetition and class
        xy = torch.einsum('ij,rjk->rik', X.T, y_centered)
        xy_sum = torch.abs(xy).sum(dim=2)
        l = xy_sum.max(dim=1)[0]

        return l


class _CoxQUT(_BaseQUTMixin):
    """QUT manager for Cox regression tasks.

    This class implements the QUT method for Cox proportional hazards models,
    where the simulation is based on bootstrap sampling of survival times.

    Args:
        lambda_qut: Custom lambda value for QUT. If None, will be computed
            automatically based on the quantile threshold.
        n_samples: Number of samples to use for QUT simulation.
        alpha: Significance level for the quantile threshold.
    """

    def __init__(
        self,
        lambda_qut: float = None,
        n_samples: int = 5000,
        alpha: float = 0.05
    ) -> None:
        """Initialize the Cox QUT manager."""
        super().__init__(lambda_qut, n_samples, alpha)

    def _simulation_method(
        self,
        X: torch.Tensor,
        target: torch.Tensor,
        n_reps: int
    ) -> torch.Tensor:
        """Simulate QUT values for Cox regression.

        Args:
            X: Input features of shape (n_samples, n_features)
            target: Target with times and events of shape (n_samples, 2)
            n_reps: Number of repetitions for simulation

        Returns:
            Simulated QUT values of shape (n_reps,)
        """
        device, dtype = X.device, X.dtype
        n_samples, _ = X.shape
        times = target[:, 0]
        events = target[:, 1]

        # Bootstrap sampling
        pair_boot_idx = torch.randint(
            0, n_samples, (n_samples, n_reps), dtype=torch.long, device=device
        )
        times_boot = times[pair_boot_idx]
        events_boot = events[pair_boot_idx]  # (n_samples, n_reps)

        # Compute cumulative statistics
        counts = torch.arange(1, n_samples + 1, dtype=dtype, device=device).unsqueeze(1)
        cumsums_mean = X.cumsum(0).div(counts)  # (n_samples, n_features)
        X_diff = X - cumsums_mean  # (n_samples, n_features)

        # Compute gradient components
        grad_nom = -(X_diff.unsqueeze(1) * events_boot.unsqueeze(2)).sum(dim=0)
        grad_denom = (
            events_boot * torch.log(counts.squeeze(1)).unsqueeze(1)
        ).sum(0).sqrt().mul(2)

        quotient = grad_nom / (grad_denom + 1e-10).unsqueeze(1)

        return quotient.abs().amax(1)


class _GumbelQUT(_BaseQUTMixin):
    """QUT manager for Gumbel tasks.

    This class implements the QUT method for Gumbel distribution problems.

    Args:
        lambda_qut: Custom lambda value for QUT. If None, will be computed
            automatically based on the quantile threshold.
        n_samples: Number of samples to use for QUT simulation.
        alpha: Significance level for the quantile threshold.
    """

    def __init__(
        self,
        lambda_qut: float = None,
        n_samples: int = 5000,
        alpha: float = 0.05
    ) -> None:
        """Initialize the Gumbel QUT manager."""
        super().__init__(lambda_qut, n_samples, alpha)

    def _simulation_method(
        self,
        X: torch.Tensor,
        target: torch.Tensor,
        n_reps: int
    ) -> torch.Tensor:
        """Simulate QUT values for Gumbel distribution.

        Args:
            X: Input features of shape (n_samples, n_features)
            target: Target values of shape (n_samples,)
            n_reps: Number of repetitions for simulation

        Returns:
            Simulated QUT values of shape (n_reps,)
        """
        device, dtype = X.device, X.dtype
        n_samples, _ = X.shape

        # Generate Gumbel samples
        # Using the standard Gumbel distribution
        uniform_samples = torch.empty(
            n_samples, n_reps, dtype=dtype, device=device
        ).uniform_(1e-10, 1 - 1e-10)

        gumbel_samples = -torch.log(-torch.log(uniform_samples))
        gumbel_centered = gumbel_samples - gumbel_samples.mean(dim=0, keepdim=True)

        # Compute X^T * y
        Xy = torch.matmul(X.T, gumbel_centered)
        Xy_inf_norm = torch.abs(Xy).amax(dim=0)
        y_norm = gumbel_centered.norm(p=2, dim=0)

        return Xy_inf_norm / y_norm

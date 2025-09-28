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
        n = X.shape[0]
        y = torch.randn(n, n_reps, device=device, dtype=dtype)
        y.sub_(y.mean(dim=0, keepdim=True))
        numer = torch.linalg.norm(X.T @ y, ord=float("inf"), dim=0)
        denom = torch.linalg.norm(y, ord=2, dim=0)
        return numer / denom

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
        self.n_classes = None

    def _compute_n_classes(self, target: torch.Tensor) -> int:
        """Compute the number of classes from target labels.

        Args:
            target: Target labels of shape (n_samples,)

        Returns:
            Number of unique classes.
        """
        self.n_classes = target.unique().numel()
        return self.n_classes

    def _compute_class_distribution(self, target: torch.Tensor) -> torch.Tensor:
        """Compute empirical class distribution from target labels.

        Args:
            target: Target labels of shape (n_samples,)

        Returns:
            Class probabilities of shape (n_classes,)
        """
        counts = torch.bincount(target, minlength=self._compute_n_classes(target)).float()
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
        n = X.shape[0]
        hat_p = self._compute_class_distribution(target).to(device=device, dtype=dtype)
        #hat_p = torch.tensor([0.5, 0.5], device=device, dtype=dtype)
        y = torch.multinomial(
            hat_p, X.shape[0] * n_reps, replacement=True
        ).reshape(n_reps, X.shape[0])

        y = torch.nn.functional.one_hot(
            y, num_classes=self.n_classes
        ).float()
        y.sub_(y.mean(dim=1, keepdim=True))

        xy = torch.einsum('ij,rjk->rik', X.T, y)
        l = xy.abs().sum(dim=2).max(dim=1)[0]

        #ybar = y_mean[..., 1].squeeze()
        #l = l / torch.sqrt(ybar * (1-ybar))
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



        Code for sorted X at each rep:

        device, dtype = X.device, X.dtype
        n_samples, n_feat = X.shape
        times = target[:, 0]
        events = target[:, 1]

        # Bootstrap sampling
        pair_boot_idx = torch.randint(
            0, n_samples, (n_samples, n_reps), dtype=torch.long, device=device
        )
        times_boot = times[pair_boot_idx]
        events_boot = events[pair_boot_idx]  # (n_samples, n_reps)

        sorted_idx = torch.argsort(times_boot, dim=0, descending=True)
        events = events_boot.gather(0, sorted_idx)
        X = X.unsqueeze(2).expand(n_samples, n_feat, n_reps).gather(0, sorted_idx.unsqueeze(1).expand(-1, n_feat, -1))

        # Compute cumulative statistics
        counts = torch.arange(1, n_samples + 1, dtype=dtype, device=device)
        X.sub_(X.cumsum(0) / counts.view(-1, 1, 1))

        # Compute gradient components
        grad_nom = torch.linalg.norm((events.unsqueeze(1) * X).sum(0), ord=float("inf"), dim=0)
        grad_denom = 2 * (events.T.matmul(torch.log(counts))).sqrt()

        quotient = grad_nom / grad_denom.add(1e-10)
        return quotient
        """
        device, dtype = X.device, X.dtype
        n_samples, _ = X.shape
        times = target[:, 0]
        events = target[:, 1]

        pair_boot_idx = torch.randint(
            0, n_samples, (n_samples, n_reps), dtype=torch.long, device=device
        )
        times_boot = times[pair_boot_idx]
        events_boot = events[pair_boot_idx]

        sorted_idx = torch.argsort(times_boot, dim=0, descending=True)
        events = events_boot.gather(0, sorted_idx)

        counts = torch.arange(1, n_samples + 1, dtype=dtype, device=device)
        X.sub_(X.cumsum(0) / counts.view(-1, 1))

        grad_nom = torch.linalg.norm(events.T @ X, ord=float("inf"), dim=1)
        grad_denom = 2 * (events.T.matmul(torch.log(counts))).sqrt()

        quotient = grad_nom / grad_denom.add(1e-10)
        return quotient


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

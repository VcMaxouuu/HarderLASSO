"""Base QUT mixin for quantile universal threshold computation.

This module provides the abstract base class for QUT (Quantile Universal Threshold)
implementations in the HarderLASSO framework.
"""

from abc import ABC, abstractmethod
from typing import Optional, List, Dict, Any, Callable, Union
import numpy as np
import torch


class _BaseQUTMixin(ABC):
    """Abstract base class for Quantile Universal Threshold (QUT) functionality.

    This mixin provides the interface and common functionality for computing
    QUT values across different task types. QUT is used to determine the
    regularization parameter lambda for feature selection.

    The QUT method uses Monte Carlo simulation to estimate the distribution
    of the test statistic under the null hypothesis (no signal), and then
    uses a quantile of this distribution as the threshold.

    Attributes:
        n_samples: Number of Monte Carlo samples for QUT computation
        alpha: Significance level for QUT computation
        lambda_qut_: The computed QUT value
        lambda_path_: The computed lambda path for regularization
        monte_carlo_samples_: Monte Carlo simulation results

    Note:
        This is an internal class and should not be used directly by end users.
    """

    def __init__(
        self,
        lambda_qut: Optional[float] = None,
        n_samples: int = 5000,
        alpha: float = 0.05
    ) -> None:
        """Initialize the QUT mixin.

        Args:
            lambda_qut: Predefined QUT value. If provided, computation is skipped.
            n_samples: Number of Monte Carlo samples for QUT computation. Default is 5000.
            alpha: Significance level for QUT computation.  Default is 0.05.
        """
        self.n_samples = n_samples
        self.alpha = alpha
        self.lambda_qut_ = lambda_qut
        self.lambda_path_: Optional[np.ndarray] = None
        self.monte_carlo_samples_: Optional[np.ndarray] = None

    @abstractmethod
    def _simulation_method(
        self,
        X: torch.Tensor,
        target: torch.Tensor,
        n_reps: int
    ) -> torch.Tensor:
        """Generate Monte Carlo simulation statistics.

        This method implements the task-specific simulation logic for generating
        the null distribution of the test statistic.

        Args:
            X: Input features of shape (n_samples, n_features)
            target: Target values (shape depends on task)
            n_reps: Number of repetitions for the simulation

        Returns:
            Simulation statistics of shape (n_reps,)

        Note:
            Subclasses must implement this method to define the simulation
            logic appropriate for their specific task type.
        """
        raise NotImplementedError("Subclasses must implement `_simulation_method`.")

    def compute_lambda_qut(
        self,
        X: torch.Tensor,
        target: torch.Tensor,
        hidden_dims: Optional[List[int]] = None,
        activation: Optional[Callable] = None
    ) -> float:
        """Compute the Quantile Universal Threshold (QUT) value.

        Args:
            X: Input features of shape (n_samples, n_features)
            target: Target values (shape depends on task)
            hidden_dims: Hidden layer dimensions of the network for scaling
            activation: Activation function used in the network for scaling

        Returns:
            The computed QUT value

        Note:
            If lambda_qut was provided in __init__, it is returned directly.
            Otherwise, Monte Carlo simulation is performed to estimate the
            (1-alpha) quantile of the null distribution.
        """
        if self.lambda_qut_ is not None:
            return self.lambda_qut_

        # Perform Monte Carlo simulation
        full_list = self._simulation_method(X, target, self.n_samples)

        # Apply neural network scaling if parameters are provided
        if hidden_dims:
            full_list = self._apply_scaling_for_networks(full_list, activation, hidden_dims)

        # Compute the quantile
        self.lambda_qut_ = torch.quantile(full_list, 1 - self.alpha).item()
        self.monte_carlo_samples_ = full_list.cpu().numpy()

        return self.lambda_qut_

    def _apply_scaling_for_networks(
        self,
        full_list: torch.Tensor,
        activation: Callable,
        hidden_dims: List[int]
    ) -> torch.Tensor:
        """Apply scaling to simulation results for neural networks.

        This method applies a scaling factor that accounts for the network
        architecture when computing the QUT value.

        Args:
            full_list: Monte Carlo simulation results
            activation: Activation function used in the network
            hidden_dims: Hidden layer dimensions of the network

        Returns:
            Scaled simulation results
        """
        if len(hidden_dims) == 1:
            pi_l = 1.0
        else:
            pi_l = np.sqrt(np.prod(hidden_dims[1:]))

        device = full_list.device
        dtype = full_list.dtype

        max_derivative = self._compute_activation_derivative_supremum(activation, device, dtype)
        sigma_diff = max_derivative ** len(hidden_dims)

        # Apply scaling
        full_list = full_list * pi_l * sigma_diff
        return full_list

    def _compute_activation_derivative_supremum(
        self,
        activation: Callable,
        device: torch.device,
        dtype: torch.dtype
        ) -> float:
        """Compute the supremum of the absolute derivative of the activation function.

        Args:
            activation: Activation function
            device: torch device
            dtype: torch dtype

        Returns:
            Supremum of |σ'(t)| over all t
        """
        # For common activation functions, we can use analytical results
        activation_name = getattr(activation, '__name__', str(activation))
        if hasattr(activation, '__class__'):
            activation_name = activation.__class__.__name__

        if activation_name in ['ReLU', 'relu', 'Softplus', 'softplus', 'PReLU', 'prelu']:
            return 1.0
        elif activation_name in ['LeakyReLU', 'leaky_relu']:
            negative_slope = getattr(activation, 'negative_slope', 0.01)
            return max(1.0, abs(negative_slope))
        elif activation_name in ['ELU', 'elu']:
            alpha = getattr(activation, 'alpha', 1.0)
            return max(1.0, alpha)
        else:
            return self._numerical_derivative_supremum(activation, dtype=dtype, device=device)


    def _numerical_derivative_supremum(
        self,
        activation: Callable,
        search_range: float = 20.0,
        num_points: int = 20000,
        device: torch.device = None,
        dtype: torch.dtype = torch.float32
    ) -> float:
        """
        Numerically compute the supremum of the absolute derivative of `activation`
        over [-search_range, search_range].

        Args:
            activation: Activation function mapping Tensor -> Tensor
            search_range: Range to search over [-search_range, search_range]
            num_points: Number of points to evaluate
            device: torch device (defaults to GPU if available)
            dtype: torch dtype

        Returns:
            Numerical approximation of sup |σ'(t)|
        """
        # pick device
        if device is None:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        t = torch.linspace(-search_range,
                            search_range,
                            num_points,
                            device=device,
                            dtype=dtype,
                            requires_grad=True)

        y = activation(t)
        grad_t, = torch.autograd.grad(
            outputs=y,
            inputs=t,
            grad_outputs=torch.ones_like(y),
            create_graph=False,
            retain_graph=False
        )
        sup = grad_t.abs().max().item()

        return sup


    def compute_lambda_path(
        self,
        X: torch.Tensor,
        target: torch.Tensor,
        path_values: Optional[List[float]] = None,
        hidden_dims: Optional[List[int]] = None,
        activation: Optional[Callable] = None
    ) -> np.ndarray:
        """Compute the lambda path for regularization.

        The lambda path consists of multiple lambda values that are fractions
        of the computed QUT value. This allows for regularization path fitting.

        Args:
            X: Input features of shape (n_samples, n_features)
            target: Target values (shape depends on task)
            path_values: Fractions of lambda_qut to use for the path.
                If None, uses a default sequence.
            hidden_dims: Hidden layer dimensions of the network for scaling
            activation: Activation function used in the network for scaling

        Returns:
            Array of lambda values for the regularization path

        Note:
            If lambda_qut is not computed, it will be computed first.
        """
        if self.lambda_qut_ is None:
            self.compute_lambda_qut(X, target, hidden_dims, activation)

        if path_values is None:
            path_values = [
                np.exp(i - 1) / (1 + np.exp(i - 1)) for i in range(5)
            ] + [1.0]

        self.lambda_path_ = self.lambda_qut_ * np.array(path_values)
        return self.lambda_path_

    def plot_simulation_distribution(
        self,
        bins: int = 50,
        figsize: tuple = (10, 6)
    ) -> None:
        """Plot the distribution of Monte Carlo simulation results.

        Args:
            bins: Number of histogram bins
            figsize: Figure size in inches

        Raises:
            ValueError: If Monte Carlo samples are not computed
        """
        if self.monte_carlo_samples_ is None:
            raise ValueError(
                "Monte Carlo samples are not computed. Call `compute_lambda_qut` first."
            )

        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=figsize)

        # Plot histogram
        ax.hist(
            self.monte_carlo_samples_,
            bins=bins,
            density=True,
            alpha=0.7,
            color='skyblue'
        )

        # Add vertical line for lambda_QUT
        ax.axvline(
            self.lambda_qut_,
            color='red',
            linestyle='--',
            label=f'λ_QUT ({1-self.alpha:.2%} quantile)'
        )

        ax.set_title('Distribution of Monte Carlo Simulation Statistics')
        ax.set_xlabel('Statistic Value')
        ax.set_ylabel('Density')
        ax.legend()

        plt.show()

    def summary(self) -> Dict[str, Any]:
        """Return summary statistics of the Monte Carlo simulations.

        Returns:
            Dictionary containing summary statistics including mean, std, min,
            max, median, lambda_qut, n_samples, alpha, and quartiles.

        Raises:
            ValueError: If Monte Carlo samples are not computed
        """
        if self.monte_carlo_samples_ is None:
            raise ValueError(
                "Monte Carlo samples are not computed. Call `compute_lambda_qut` first."
            )

        samples = self.monte_carlo_samples_

        summary = {
            'mean': samples.mean(),
            'std': samples.std(),
            'min': samples.min(),
            'max': samples.max(),
            'median': np.median(samples),
            'lambda_qut': self.lambda_qut_,
            'n_samples': self.n_samples,
            'alpha': self.alpha
        }

        # Add quartiles
        q1, q3 = np.quantile(samples, [0.25, 0.75])
        summary['q1'] = q1
        summary['q3'] = q3

        return summary

    def plot_lambda_path_distribution(self, figsize: tuple = (12, 6)) -> None:
        """Visualize the lambda path distribution relative to simulation results.

        Args:
            figsize: Figure size in inches

        Raises:
            ValueError: If lambda path or Monte Carlo samples are not computed
        """
        if self.lambda_path_ is None or self.monte_carlo_samples_ is None:
            raise ValueError(
                "Lambda path and Monte Carlo samples must be computed first."
            )

        import matplotlib.pyplot as plt

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)

        # Plot 1: Lambda path points
        ax1.plot(
            range(len(self.lambda_path_)),
            self.lambda_path_,
            'o-',
            color='skyblue'
        )
        ax1.set_title('Lambda Path Values')
        ax1.set_xlabel('Path Point Index')
        ax1.set_ylabel('Lambda Value')

        # Plot 2: Lambda path relative to distribution
        ax2.hist(
            self.monte_carlo_samples_,
            bins=50,
            density=True,
            alpha=0.7,
            color='skyblue'
        )

        # Add vertical lines for each lambda path value
        for i, lambda_val in enumerate(self.lambda_path_):
            ax2.axvline(
                lambda_val,
                color=f'C{i}',
                linestyle='--',
                label=f"λ_{i}"
            )

        ax2.set_title('Lambda Path vs. Simulation Distribution')
        ax2.set_xlabel('Statistic Value')
        ax2.set_ylabel('Density')
        ax2.legend()

        plt.tight_layout()
        plt.show()

    def get_qut_params(self) -> Dict[str, Any]:
        """Get QUT parameters for serialization.

        Returns:
            Dictionary of QUT parameters
        """
        return {
            'n_samples': self.n_samples,
            'alpha': self.alpha,
            'lambda_qut_': self.lambda_qut_,
            'lambda_path_': self.lambda_path_,
            'monte_carlo_samples_': self.monte_carlo_samples_
        }

    def set_qut_params(self, params: Dict[str, Any]) -> None:
        """Set QUT parameters from deserialization.

        Args:
            params: Dictionary of QUT parameters
        """
        self.n_samples = params.get('n_samples', 5000)
        self.alpha = params.get('alpha', 0.05)
        self.lambda_qut_ = params.get('lambda_qut_')
        self.lambda_path_ = params.get('lambda_path_')
        self.monte_carlo_samples_ = params.get('monte_carlo_samples_')

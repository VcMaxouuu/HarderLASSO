from abc import ABC, abstractmethod
import torch
import numpy as np

def function_derivative(func, u):
    if not u.requires_grad:
        raise ValueError("Input tensor `u` must have `requires_grad=True`.")
    if u.grad is not None:
        u.grad.zero_()
    y = func(u)
    y.backward()
    derivative = u.grad.item()
    u.grad.zero_()
    return derivative

class QUTMixin(ABC):
    """
    A mixin class that provides Quantile Universal Threshold (QUT) functionality.

    This mixin can be used with different task-specific classes (e.g., regression, classification)
    to provide QUT computation capabilities.

    Attributes:
        n_samples (int): Number of Monte Carlo samples for QUT computation.
        alpha (float): Significance level for QUT computation (default is 0.05).
        lambda_qut (float or None): The computed QUT value.
        lambda_path (np.ndarray or None): The computed lambda path for regularization.
        monte_carlo_samples (np.ndarray or None): Monte Carlo simulation results.
    """
    def __init__(self, lambda_qut=None, n_samples: int=5000, alpha: float=0.05):
        """
        Initializes the QUTMixin.

        Args:
            lambda_qut (float, optional): Predefined QUT value. Defaults to None.
            n_samples (int, optional): Number of Monte Carlo samples. Defaults to 5000.
            alpha (float, optional): Significance level for QUT computation. Defaults to 0.05.
        """
        self.n_samples = n_samples
        self.alpha = alpha
        self.lambda_qut_ = lambda_qut
        self.lambda_path_ = None
        self.monte_carlo_samples_ = None

    @abstractmethod
    def _simulation_method(self, X: torch.Tensor, target: torch.Tensor, n_reps: int) -> torch.Tensor:
        """
        Abstract method for generating Monte Carlo simulation statistics.

        Subclasses must implement this method to define the simulation logic.

        Args:
            X (torch.Tensor): Input features as a PyTorch tensor.
            target (torch.Tensor): Target values as a PyTorch tensor.
            n_reps (int): Number of repetitions for the simulation.

        Returns:
            torch.Tensor: A tensor containing the simulation statistics.
        """
        raise NotImplementedError("Subclasses must implement `_simulation_method`.")

    def compute_lambda_qut(self, X, target, hidden_dims=None, activation=None):
        """
        Computes the Quantile Universal Threshold (QUT) value.

        Args:
            X (torch.Tensor): Input features as a PyTorch tensor.
            target (torch.Tensor): Target values as a PyTorch tensor.
            hidden_dims (list[int], optional): Hidden layer dimensions of the network. Defaults to None.
            activation (callable, optional): Activation function used in the network. Defaults to None.

        Returns:
            float: The computed QUT value.
        """
        if self.lambda_qut_ is not None:
            return self.lambda_qut_

        full_list = self._simulation_method(X, target, self.n_samples)
        if hidden_dims:
            full_list = self._apply_scaling_for_networks(full_list, activation, hidden_dims)

        self.lambda_qut_ = torch.quantile(full_list, 1 - self.alpha).item()
        self.monte_carlo_samples_ = full_list.cpu().numpy()

        return self.lambda_qut_

    def _apply_scaling_for_networks(self, full_list, activation, hidden_dims):
        """
        Applies scaling to the Monte Carlo simulation results for neural networks.

        Args:
            full_list (torch.Tensor): Monte Carlo simulation results.
            activation (callable): Activation function used in the network.
            hidden_dims (list[int]): Hidden layer dimensions of the network.

        Returns:
            torch.Tensor: Scaled simulation results.
        """
        if len(hidden_dims) == 1:
            pi_l = 1.0
        else:
            pi_l = np.sqrt(np.prod(hidden_dims[1:]))

        u = torch.tensor(0.0, dtype=full_list.dtype, device=full_list.device, requires_grad=True)
        sigma_diff = function_derivative(activation, u) ** len(hidden_dims)
        full_list = full_list * pi_l * sigma_diff
        return full_list

    def compute_lambda_path(self, X, target, path_values=None, hidden_dims=None, activation=None):
        """
        Computes the lambda path for regularization.

        Args:
            X (torch.Tensor): Input features as a PyTorch tensor.
            target (torch.Tensor): Target values as a PyTorch tensor.
            path_values (list[float], optional): Values for the lambda path. Defaults to a predefined list.
            hidden_dims (list[int], optional): Hidden layer dimensions of the network. Defaults to None.
            activation (callable, optional): Activation function used in the network. Defaults to None.

        Returns:
            np.ndarray: The computed lambda path.
        """
        if self.lambda_qut_ is None:
            self.compute_lambda_qut(X, target, hidden_dims, activation)

        if path_values is None:
            path_values = [np.exp(i - 1) / (1 + np.exp(i - 1)) for i in range(5)] + [1.0]

        self.lambda_path_ = self.lambda_qut_ * np.array(path_values)
        return self.lambda_path_


    def plot_simulation_distribution(self, bins=50, figsize=(10, 6)):
        """
        Plots the distribution of Monte Carlo simulation results with the computed lambda_QUT.

        Args:
            bins (int, optional): Number of histogram bins. Defaults to 50.
            figsize (tuple[int, int], optional): Figure size in inches. Defaults to (10, 6).

        Raises:
            ValueError: If Monte Carlo samples are not computed.
        """
        if self.monte_carlo_samples_ is None:
            raise ValueError("Monte Carlo samples are not computed. Call `compute_lambda_qut` first.")

        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=figsize)

        # Plot histogram
        ax.hist(self.monte_carlo_samples_, bins=bins, density=True, alpha=0.7, color='skyblue')

        # Add vertical line for lambda_QUT
        ax.axvline(self.lambda_qut_, color='red', linestyle='--',
                  label=f'λ_QUT ({1-self.alpha:.2%} quantile)')

        ax.set_title('Distribution of Monte Carlo Simulation Statistics')
        ax.set_xlabel('Statistic Value')
        ax.set_ylabel('Density')
        ax.legend()

        plt.show()

    def summary(self):
        """
        Returns summary statistics of the Monte Carlo simulations containing
        mean, std, min, max, median, lambda_qut, n_samples, alpha, and quartiles.

        Returns:
            dict: A dictionary containing summary statistics of the Monte Carlo simulations.

        Raises:
            ValueError: If Monte Carlo samples are not computed.
        """
        if self.monte_carlo_samples_ is None:
            raise ValueError("Monte Carlo samples are not computed. Call `compute_lambda_qut` first.")

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

    def plot_lambda_path_distribution(self, figsize=(12, 6)):
        """
        Visualizes the lambda path distribution relative to the simulation results.

        Args:
            figsize (tuple[int, int], optional): Figure size in inches. Defaults to (12, 6).

        Raises:
            ValueError: If lambda path or Monte Carlo samples are not computed.
        """
        if self.lambda_path_ is None or self.monte_carlo_samples_ is None:
            raise ValueError("Monte Carlo samples are not computed. Call `compute_lambda_qut` first.")

        import matplotlib.pyplot as plt

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)

        # Plot 1: Lambda path points
        ax1.plot(range(len(self.lambda_path_)), self.lambda_path_,
                'o-', color='skyblue')
        ax1.set_title('Lambda Path Values')
        ax1.set_xlabel('Path Point Index')
        ax1.set_ylabel('Lambda Value')

        # Plot 2: Lambda path relative to distribution
        ax2.hist(self.monte_carlo_samples_, bins=50, density=True, alpha=0.7, color='skyblue')

        # Add vertical lines for each lambda path value
        for i, lambda_val in enumerate(self.lambda_path_):
            ax2.axvline(lambda_val, color=f'C{i}',
                       linestyle='--', label=f"λ_{i}")

        ax2.set_title('Lambda Path vs. Simulation Distribution')
        ax2.set_xlabel('Statistic Value')
        ax2.set_ylabel('Density')
        ax2.legend()

        plt.tight_layout()
        plt.show()

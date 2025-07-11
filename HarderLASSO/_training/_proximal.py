"""
Proximal operators for HarderLASSO penalty functions.

This module provides proximal operators for different penalty functions
used in sparse optimization problems.
"""

import torch
import numpy as np
from scipy.optimize import root_scalar, newton
import warnings
from typing import Union, Literal
from abc import ABC, abstractmethod


class _BaseProximalOperator(ABC):
    """
    Base class for proximal operators.

    A proximal operator solves:
        prox_g(u) = argmin_x { (1/2)||x - u||^2 + g(x) }

    where g(x) is the penalty function.
    """

    @abstractmethod
    def apply(self, u: torch.Tensor, step_size: float) -> torch.Tensor:
        """
        Apply the proximal operator.

        Parameters
        ----------
        u : torch.Tensor
            Input tensor.
        step_size : float
            Step size parameter.

        Returns
        -------
        torch.Tensor
            Result of applying the proximal operator.
        """
        pass


class _LassoProximalOperator(_BaseProximalOperator):
    """
    Proximal operator for LASSO (L1) penalty.

    The LASSO penalty is g(x) = lambda * ||x||_1.
    The proximal operator is the soft-thresholding operator.

    Parameters
    ----------
    lambda_ : float
        Regularization parameter.

    Examples
    --------
    >>> prox_op = _LassoProximalOperator(lambda_=0.1)
    >>> result = prox_op.apply(torch.randn(10), step_size=0.01)
    """

    def __init__(self, lambda_: float):
        """Initialize the LASSO proximal operator."""
        if lambda_ < 0:
            raise ValueError("Lambda must be non-negative.")
        self.lambda_ = lambda_

    def apply(self, u: torch.Tensor, step_size: float) -> torch.Tensor:
        """
        Apply soft-thresholding (LASSO proximal operator).

        Parameters
        ----------
        u : torch.Tensor
            Input tensor.
        step_size : float
            Step size parameter.

        Returns
        -------
        torch.Tensor
            Soft-thresholded tensor.
        """
        threshold = self.lambda_ * step_size

        if threshold == 0:
            return u

        return torch.sign(u) * torch.relu(u.abs() - threshold)

    def __repr__(self) -> str:
        return f"_LassoProximalOperator(lambda_={self.lambda_})"


class _SCADProximalOperator(_BaseProximalOperator):
    """
    Proximal operator for SCAD (Smoothly Clipped Absolute Deviation) penalty.

    The SCAD penalty is a non-convex penalty that provides unbiased estimates
    for large coefficients while maintaining sparsity.

    Parameters
    ----------
    lambda_ : float
        Regularization parameter.
    a : float, default=3.7
        Concavity parameter.

    Examples
    --------
    >>> prox_op = _SCADProximalOperator(lambda_=0.1, a=3.7)
    >>> result = prox_op.apply(torch.randn(10), step_size=0.01)
    """

    def __init__(self, lambda_: float, a: float = 3.7):
        """Initialize the SCAD proximal operator."""
        if lambda_ < 0:
            raise ValueError("Lambda must be non-negative.")
        if a <= 2:
            raise ValueError("SCAD parameter 'a' must be greater than 2.")

        self.lambda_ = lambda_
        self.a = a

    def apply(self, u: torch.Tensor, step_size: float) -> torch.Tensor:
        """
        Apply SCAD proximal operator.

        Parameters
        ----------
        u : torch.Tensor
            Input tensor.
        step_size : float
            Step size parameter.

        Returns
        -------
        torch.Tensor
            SCAD-thresholded tensor.
        """
        threshold = self.lambda_ * step_size

        if threshold == 0:
            return u

        result = torch.zeros_like(u)
        abs_u = u.abs()

        # Region 1: |u| <= threshold -> set to 0
        mask1 = abs_u <= threshold
        result[mask1] = 0

        # Region 2: threshold < |u| <= 2*threshold -> soft thresholding
        mask2 = (abs_u > threshold) & (abs_u <= 2 * threshold)
        result[mask2] = torch.sign(u[mask2]) * (abs_u[mask2] - threshold)

        # Region 3: 2*threshold < |u| <= a*threshold -> SCAD correction
        mask3 = (abs_u > 2 * threshold) & (abs_u <= self.a * threshold)
        numerator = (self.a - 1) * abs_u[mask3] - self.a * threshold
        denominator = self.a - 2
        result[mask3] = torch.sign(u[mask3]) * numerator / denominator

        # Region 4: |u| > a*threshold -> no shrinkage
        mask4 = abs_u > self.a * threshold
        result[mask4] = u[mask4]

        return result

    def __repr__(self) -> str:
        return f"_SCADProximalOperator(lambda_={self.lambda_}, a={self.a})"


class _HarderProximalOperator(_BaseProximalOperator):
    """
    Proximal operator for Harder penalty.

    The Harder penalty is a non-convex penalty that interpolates between
    LASSO and ridge regression based on the parameter nu.

    Parameters
    ----------
    lambda_ : float
        Regularization parameter.
    nu : float
        Interpolation parameter between 0 and 1.

    Examples
    --------
    >>> prox_op = _HarderProximalOperator(lambda_=0.1, nu=0.5)
    >>> result = prox_op.apply(torch.randn(10), step_size=0.01)
    """

    def __init__(self, lambda_: float, nu: float):
        """Initialize the Harder proximal operator."""
        if lambda_ < 0:
            raise ValueError("Lambda must be non-negative.")
        if not 0 < nu < 1:
            raise ValueError("Nu must be between 0 and 1.")

        self.lambda_ = lambda_
        self.nu = nu

    def _find_threshold(self, regularization: float) -> float:
        """
        Find the threshold value for the Harder penalty.

        Parameters
        ----------
        regularization : float
            Regularization strength (lambda * step_size).

        Returns
        -------
        float
            Threshold value.
        """
        def func(kappa):
            return (kappa**self.nu + 2*kappa + kappa**(2 - self.nu) +
                   2*regularization*(self.nu - 1))

        try:
            bracket = [0, regularization * (1 - self.nu) / 2]
            solution = root_scalar(func, bracket=bracket, method='brentq')
            kappa = solution.root

            # Compute threshold
            threshold = kappa / 2 + regularization / (1 + kappa**(1 - self.nu))
            return threshold

        except (ValueError, RuntimeError):
            # Fallback to simple threshold
            warnings.warn(
                "Could not find exact threshold for Harder penalty. "
                "Using approximation.",
                RuntimeWarning
            )
            return regularization

    def _solve_nonzero(self, u_val: float, regularization: float) -> float:
        """
        Solve for non-zero solution in Harder penalty.

        Parameters
        ----------
        u_val : float
            Input value.
        regularization : float
            Regularization strength.

        Returns
        -------
        float
            Solution value.
        """
        nu = self.nu

        def func(x, y, reg):
            return x - abs(y) + reg * (1 + nu * x**(1 - nu)) / (1 + x**(1 - nu))**2

        def fprime(x, y, reg):
            numerator = (1 - nu) * (2 + nu * x**(1 - nu) - nu)
            denominator = x**nu * (1 + x**(1 - nu))**3
            return 1 - reg * numerator / denominator

        try:
            root = newton(
                func,
                x0=abs(u_val),
                fprime=fprime,
                args=(u_val, regularization),
                maxiter=500,
                tol=1e-5
            )
            return root * np.sign(u_val)

        except RuntimeError:
            warnings.warn(
                "Newton-Raphson did not converge for Harder penalty. "
                "Using soft-thresholding approximation.",
                RuntimeWarning
            )
            # Fallback to soft thresholding
            threshold = regularization
            return np.sign(u_val) * max(0, abs(u_val) - threshold)

    def apply(self, u: torch.Tensor, step_size: float) -> torch.Tensor:
        """
        Apply Harder proximal operator.

        Parameters
        ----------
        u : torch.Tensor
            Input tensor.
        step_size : float
            Step size parameter.

        Returns
        -------
        torch.Tensor
            Harder-thresholded tensor.
        """
        regularization = self.lambda_ * step_size

        if regularization == 0:
            return u

        result = torch.zeros_like(u)
        abs_u = u.abs()

        # Find threshold
        threshold = self._find_threshold(regularization)

        # Only process values above threshold
        mask = abs_u > threshold

        if not mask.any():
            return result

        # Solve for non-zero values
        u_selected = u[mask].cpu().numpy()
        solutions = torch.tensor(
            self._solve_nonzero(u_selected, regularization),
            dtype=u.dtype,
            device=u.device
        )
        result[mask] = solutions

        return result

    def __repr__(self) -> str:
        return f"_HarderProximalOperator(lambda_={self.lambda_}, nu={self.nu})"


def _create_proximal_operator(
    penalty: Literal['lasso', 'scad', 'harder'] = 'lasso',
    lambda_: float = None,
    **kwargs
) -> _BaseProximalOperator:
    """
    Factory function to create proximal operators.

    Parameters
    ----------
    penalty : {'lasso', 'scad', 'harder'}, default='lasso'
        Type of penalty.
    lambda_ : float
        Regularization parameter.
    **kwargs
        Additional parameters specific to each penalty.

    Returns
    -------
    _BaseProximalOperator
        Configured proximal operator.

    Raises
    ------
    ValueError
        If penalty type is unknown or required parameters are missing.

    Examples
    --------
    >>> prox_op = _create_proximal_operator('lasso', lambda_=0.1)
    >>> prox_op = _create_proximal_operator('scad', lambda_=0.1, a=3.7)
    >>> prox_op = _create_proximal_operator('harder', lambda_=0.1, nu=0.5)
    """
    # Handle legacy parameter names
    if lambda_ is None:
        lambda_ = kwargs.get('lambda', kwargs.get('lambda_'))

    if lambda_ is None:
        raise ValueError("Regularization parameter 'lambda_' is required.")

    penalty = penalty.lower()

    if penalty == 'lasso':
        return _LassoProximalOperator(lambda_=lambda_)

    elif penalty == 'scad':
        a = kwargs.get('a', 3.7)
        return _SCADProximalOperator(lambda_=lambda_, a=a)

    elif penalty == 'harder':
        nu = kwargs.get('nu')
        if nu is None:
            raise ValueError("Parameter 'nu' is required for Harder penalty.")
        return _HarderProximalOperator(lambda_=lambda_, nu=nu)

    else:
        raise ValueError(
            f"Unknown penalty '{penalty}'. "
            "Supported penalties are: 'lasso', 'scad', 'harder'."
        )

import torch
import numpy as np
from scipy.optimize import root_scalar, newton
import warnings

class BasePenalty:
    """
    Base class for penalties used in optimization.
    Subclasses should implement the `value` and `proximal` methods.
    """
    def __init__(self, **kwargs):
        """Initialize penalty with default parameters."""
        self.params = kwargs.copy()
        self._last_value = None  # Store the last computed penalty value

    def update_params(self, **kwargs):
        """Update penalty parameters.

        Parameters
        ----------
        **kwargs : dict
            Parameter updates to apply.
        """
        self.params.update(kwargs)

    @property
    def last_value(self):
        """Get the last computed penalty value.

        Returns
        -------
        float or torch.Tensor or None
            The last computed penalty value, or None if no value has been computed yet.
        """
        return self._last_value

    def value(self, parameter, **kwargs):
        raise NotImplementedError("Subclasses must implement this method.")

    def proximal(self, parameter, **kwargs):
        raise NotImplementedError("Subclasses must implement this method.")

    def __str__(self):
        return self.__class__.__name__

    def __repr__(self):
        return f"{self.__class__.__name__}()"

class L1Penalty(BasePenalty):
    """
    Class to compute the L1 penalty for a given parameter.
    """
    def __init__(self, lambda_=0.0):
        """Initialize L1 penalty.

        Parameters
        ----------
        lambda_ : float, default=0.0
            L1 regularization parameter.
        """
        super().__init__(lambda_=lambda_)

    def value(self, parameter, **kwargs):
        # Use stored parameter if not provided in kwargs
        lambda_ = kwargs.get('lambda_', self.params.get('lambda_', 0))
        penalty_value = lambda_ * torch.sum(parameter.abs())
        self._last_value = penalty_value  # Store the computed value
        return penalty_value

    def proximal(self, parameter, **kwargs):
        # Use stored parameters if not provided in kwargs
        lambda_ = kwargs.get('lambda_', self.params.get('lambda_', 0))
        lr = kwargs.get('lr', 0.01)

        lr_lambda = lambda_ * lr

        if lr_lambda == 0:
            return parameter
        return torch.sign(parameter) * torch.relu(parameter.abs() - lr_lambda)

    def __str__(self):
        return "L1 Penalty"

    def __repr__(self):
        return f"L1Penalty(lambda_={self.params.get('lambda_', 0.0)})"

class HarderPenalty(BasePenalty):
    """
    Class to compute the HarderLASSO penalty for a given parameter.
    """
    def __init__(self, lambda_=0.0, nu=0.1):
        """Initialize Harder penalty.

        Parameters
        ----------
        lambda_ : float, default=0.0
            Regularization parameter.
        nu : float, default=0.1
            Harder penalty shape parameter.
        """
        super().__init__(lambda_=lambda_, nu=nu)

    def value(self, parameter, **kwargs):
        # Use stored parameters if not provided in kwargs
        lambda_ = kwargs.get('lambda_', self.params.get('lambda_', 0))
        nu = kwargs.get('nu', self.params.get('nu', 0.1))

        eps = 1e-10
        abs_p = parameter.abs() + eps
        pow_term = torch.pow(abs_p, 1 - nu)
        penalty_value = lambda_ * (abs_p / (1 + pow_term)).sum()
        self._last_value = penalty_value  # Store the computed value
        return penalty_value

    def _find_threshold(self, lr_lambda: float, nu: float) -> float:
        """
        Find the threshold value for the Harder penalty.

        Parameters
        ----------
        lr_lambda : float
            lambda * step_size.
        nu : float
            Harder penalty parameter.

        Returns
        -------
        float
            Threshold value.
        """
        def func(kappa):
            return (kappa**nu + 2*kappa + kappa**(2 - nu) + 2*lr_lambda*(nu - 1))

        try:
            bracket = [0, lr_lambda * (1 - nu) / 2]
            solution = root_scalar(func, bracket=bracket, method='brentq')
            kappa = solution.root

            # Compute threshold
            threshold = kappa / 2 + lr_lambda / (1 + kappa**(1 - nu))
            return threshold

        except (ValueError, RuntimeError):
            # Fallback to simple threshold
            warnings.warn(
                "Could not find exact threshold for Harder penalty. "
                "Using approximation.",
                RuntimeWarning
            )
            return lr_lambda

    def _solve_nonzero(self, u_val: float, lr_lambda: float, nu: float) -> float:
        """
        Solve for non-zero solution in Harder penalty.

        Parameters
        ----------
        u_val : float
            Input value.
        lr_lambda : float
            lambda * step_size.
        nu : float
            Harder penalty parameter.

        Returns
        -------
        float
            Solution value.
        """

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
                args=(u_val, lr_lambda),
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
            return np.sign(u_val) * max(0, abs(u_val) - lr_lambda)

    def proximal(self, parameter, **kwargs):
        # Use stored parameters if not provided in kwargs
        lambda_ = kwargs.get('lambda_', self.params.get('lambda_', 0))
        lr = kwargs.get('lr', 0.01)
        nu = kwargs.get('nu', self.params.get('nu', 0.1))

        lr_lambda = lambda_ * lr

        if lr_lambda == 0:
            return parameter

        result = torch.zeros_like(parameter)
        abs_u = parameter.abs()

        # Find threshold
        threshold = self._find_threshold(lr_lambda, nu)

        # Only process values above threshold
        mask = abs_u > threshold

        if not mask.any():
            return result

        # Solve for non-zero values
        u_selected = parameter[mask].cpu().numpy()
        solutions = torch.tensor(
            self._solve_nonzero(u_selected, lr_lambda, nu),
            dtype=parameter.dtype,
            device=parameter.device
        )
        result[mask] = solutions

        return result

    def __str__(self):
        return "Harder Penalty"

    def __repr__(self):
        return f"HarderPenalty(lambda_={self.params.get('lambda_', 0.0)}, nu={self.params.get('nu', 0.1)})"

class SCADPenalty(BasePenalty):
    """
    Class to compute the SCAD penalty for a given parameter.
    """
    def __init__(self, lambda_=0.0, a=3.7):
        """Initialize SCAD penalty.

        Parameters
        ----------
        lambda_ : float, default=0.0
            Regularization parameter.
        a : float, default=3.7
            SCAD shape parameter.
        """
        super().__init__(lambda_=lambda_, a=a)

    def value(self, parameter, **kwargs):
        # Use stored parameters if not provided in kwargs
        lambda_ = kwargs.get('lambda_', self.params.get('lambda_', 0))
        a = kwargs.get('a', self.params.get('a', 3.7))

        abs_p = parameter.abs()
        reg_loss = 0.0

        # SCAD penalty regions
        mask1 = abs_p <= lambda_
        mask2 = (abs_p > lambda_) & (abs_p <= a * lambda_)
        mask3 = abs_p > a * lambda_

        linear_part = (lambda_ * abs_p[mask1]).sum()

        numer = 2 * a * lambda_ * abs_p[mask2] - parameter[mask2]**2 - lambda_**2
        quadratic_part = (numer / (2 * (a - 1))).sum()

        constant_part = ((lambda_**2 * (a + 1) / 2) * mask3).sum()

        reg_loss = linear_part + quadratic_part + constant_part
        self._last_value = reg_loss  # Store the computed value
        return reg_loss

    def proximal(self, parameter, **kwargs):
        # Use stored parameters if not provided in kwargs
        lambda_ = kwargs.get('lambda_', self.params.get('lambda_', 0))
        lr = kwargs.get('lr', 0.01)
        a = kwargs.get('a', self.params.get('a', 3.7))

        lr_lambda = lambda_ * lr

        if lr_lambda == 0:
            return parameter

        result = torch.zeros_like(parameter)
        abs_u = parameter.abs()

        mask1 = (abs_u > lr_lambda) & (abs_u <= 2 * lr_lambda)
        result[mask1] = (torch.sign(parameter[mask1]) * (abs_u[mask1] - lr_lambda))

        mask2 = (abs_u > 2 * lr_lambda) & (abs_u <= a * lr_lambda)
        result[mask2] = torch.sign(parameter[mask2]) * ((a - 1) * abs_u[mask2] - a * lr_lambda) / (a - 2)

        mask3 = abs_u > a * lr_lambda
        result[mask3] = parameter[mask3]

        return result

    def __str__(self):
        return "SCAD Penalty"

    def __repr__(self):
        return f"SCADPenalty(lambda_={self.params.get('lambda_', 0.0)}, a={self.params.get('a', 3.7)})"


class MCPenalty(BasePenalty):
    """
    Class to compute the MCP penalty for a given parameter.
    """
    def __init__(self, lambda_=0.0, gamma=3.0):
        """Initialize MCP penalty.

        Parameters
        ----------
        lambda_ : float, default=0.0
            Regularization parameter.
        gamma : float, default=3.0
            MCP shape parameter.
        """
        super().__init__(lambda_=lambda_, gamma=gamma)

    def value(self, parameter, **kwargs):
        # Use stored parameters if not provided in kwargs
        lambda_ = kwargs.get('lambda_', self.params.get('lambda_', 0))
        gamma = kwargs.get('gamma', self.params.get('gamma', 3.0))

        abs_p = parameter.abs()
        reg_loss = 0.0

        mask = abs_p <= lambda_ * gamma

        part1 = (lambda_ * abs_p[mask] - (parameter[mask]**2)/(2*gamma)).sum()
        part2 = (gamma * lambda_**2 / 2) * (~mask).sum()

        reg_loss = part1 + part2
        self._last_value = reg_loss  # Store the computed value
        return reg_loss

    def proximal(self, parameter, **kwargs):
        # Use stored parameters if not provided in kwargs
        lambda_ = kwargs.get('lambda_', self.params.get('lambda_', 0))
        lr = kwargs.get('lr', 0.01)
        gamma = kwargs.get('gamma', self.params.get('gamma', 3.0))

        lr_lambda = lambda_ * lr

        if lr_lambda == 0:
            return parameter

        result = torch.zeros_like(parameter)
        abs_u = parameter.abs()

        # Region 1: |u| > threshold * gamma -> no shrinkage
        mask1 = abs_u > lr_lambda * gamma
        result[mask1] = parameter[mask1]

        # Region 2: threshold < |u| <= gamma*threshold
        mask2 = (abs_u > lr_lambda) & (abs_u <= gamma * lr_lambda)
        result[mask2] = (torch.sign(parameter[mask2]) * (abs_u[mask2] - lr_lambda))/(1 - 1/gamma)

        return result

    def __str__(self):
        return "MC Penalty"

    def __repr__(self):
        return f"MCPenalty(lambda_={self.params.get('lambda_', 0.0)}, gamma={self.params.get('gamma', 3.0)})"

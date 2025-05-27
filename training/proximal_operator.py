import torch
import numpy as np
from scipy.optimize import root_scalar, newton
import warnings

class ProximalOperator:
    def __init__(self, lambda_, nu):
        self.lambda_ = lambda_
        self.nu = nu
        self.epsi = 1

    def find_thresh(self, reg):
        def func(kappa):
            return (self.epsi**2)*(kappa**self.nu) + 2*self.epsi*kappa + kappa**(2 - self.nu) + 2*reg*(self.nu - 1)

        bracket = [0, reg * (1 - self.nu) / (2*self.epsi)]
        solution = root_scalar(func, bracket=bracket)
        kappa = solution.root
        phi = kappa / 2 + reg / (self.epsi + kappa**(1 - self.nu))
        return phi

    def nonzero_sol(self, u, reg):
        nu = self.nu
        def func(x, y, reg):
            return x - np.abs(y) + reg * (self.epsi + nu * x**(1 - nu)) / (self.epsi + x**(1 - nu))**2

        def fprime(x, y, reg):
            numerator = (1 - nu) * (2 + nu * x**(1 - nu) - nu)
            denominator = x**nu * (1 + x**(1 - nu))**3
            return 1 - reg * (numerator / denominator)

        try:
            root = newton(
                func,
                x0=np.abs(u),
                fprime=fprime,
                args=(u, reg),
                maxiter=500,
                tol=1e-5
            )
        except RuntimeError:
            warnings.warn(
                "Newton-Raphson did not converge. Returning absolute value of input.",
                RuntimeWarning
            )
            root = np.abs(u)
        return root * np.sign(u)

    def apply(self, u, lr):
        reg = self.lambda_ * lr
        # If regularization is zero, no shrinkage is applied.
        if reg==0:
            return u

        # If nu is None, perform basic soft thresholding.
        if self.nu is None:
            return torch.sign(u) * torch.relu(u.abs() - reg)

        # If nu is not None, solve equation.
        sol = torch.zeros_like(u)
        abs_u = u.abs()
        phi = self.find_thresh(reg)

        ind = abs_u > phi

        if not ind.any():
            return sol

        u_selected = u[ind].cpu().numpy()
        sol_values = torch.tensor(
            self.nonzero_sol(u_selected, reg),
            dtype=u.dtype,
            device=u.device
        )
        sol[ind] = sol_values
        return sol

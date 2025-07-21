import torch
import numpy as np
from torch.optim.optimizer import Optimizer
from typing import Callable, Optional, Any, Dict


class _FISTAOptimizer(Optimizer):
    """
    FISTA optimizer with backtracking line search and momentum.

    This optimizer implements the Fast Iterative Shrinkage-Thresholding Algorithm
    for solving optimization problems of the form:

        minimize F(x) = f(x) + g(x)

    where f(x) is smooth (gradient available) and g(x) is handled by a proximal operator.

    FISTA uses momentum to accelerate convergence compared to ISTA.
    """

    def __init__(
        self,
        params,
        penalty,
        lr: float = 0.1,
        lambda_: float = 0.0,
        lr_decay: float = 0.5,
        min_lr: float = 1e-7,
    ):
        """Initialize the FISTA optimizer."""
        if lr <= 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if lr_decay <= 0.0 or lr_decay >= 1.0:
            raise ValueError(f"Invalid learning rate decay: {lr_decay}")
        if min_lr <= 0.0:
            raise ValueError(f"Invalid minimum learning rate: {min_lr}")
        if not hasattr(penalty, 'proximal'):
            raise ValueError("Penalty function must implement a 'proximal' method for FISTA")

        defaults = dict(
            penalty=penalty,
            lr=lr,
            lambda_=lambda_,
            lr_decay=lr_decay,
            min_lr=min_lr,
        )
        super().__init__(params, defaults)

        # Initialize state for each parameter group
        for group in self.param_groups:
            group['step_count'] = 0
            group['converged'] = False
            group['line_search_failed'] = False  # Track line search failures

            # Initialize momentum state for each parameter
            for p in group['params']:
                param_state = self.state[p]
                param_state['momentum_buffer'] = p.data.clone()  # y_k in FISTA
                param_state['t'] = 1.0  # momentum coefficient

    def step(self, closure: Callable) -> torch.Tensor:
        """
        Perform a single FISTA optimization step.

        closure should return (total_loss, smooth_loss) when called with backward=True,
        and return (total_loss, _) when backward=False.

        Returns the total loss at the current iterate (x_k+1) for convergence checking.
        """
        if closure is None:
            raise ValueError("FISTA requires a closure that returns loss and calls backward().")

        # Iterate over parameter groups
        for group in self.param_groups:
            self._step_group(group, closure)

        # Return total loss at current iterate for convergence checking
        return closure(backward=False)

    @torch.no_grad()
    def _set_parameters(self, params, values):
        for p, v in zip(params, values):
            p.data.copy_(v)

    def _step_group(self, group: Dict[str, Any], closure: Callable) -> None:
        """Perform FISTA step for a parameter group."""
        params = group['params']
        lr = group['lr']
        lambda_ = group['lambda_']
        lr_decay = group['lr_decay']
        min_lr = group['min_lr']
        penalty = group['penalty']

        # Store current iterate x_k
        x_k = [p.data.clone() for p in params]

        # Set parameters to momentum point y_k for gradient computation
        momentum_points = []
        for p in params:
            param_state = self.state[p]
            y_k = param_state['momentum_buffer']
            momentum_points.append(y_k.clone())
            p.data.copy_(y_k)

        # Evaluate gradient at momentum point y_k
        total_loss, smooth_loss = closure(backward=True)
        grads = [p.grad.clone() if p.grad is not None else torch.zeros_like(p) for p in params]

        # Backtracking line search starting from momentum point
        current_lr = lr
        new_x = None
        line_search_success = False

        while current_lr >= min_lr:
            # Proximal gradient step from momentum point: x_k+1 = prox(y_k - lr * grad_f(y_k))
            new_x = []
            for y_k, g_k in zip(momentum_points, grads):
                v = y_k - current_lr * g_k
                new_x.append(penalty.proximal(parameter=v, lr=current_lr, lambda_=lambda_))

            # Set parameters to new iterate and evaluate
            self._set_parameters(params, new_x)
            new_total, new_smooth = closure(backward=False)

            # Armijo condition at momentum point: f(x_k+1) <= f(y_k) + <grad_f(y_k), x_k+1-y_k> + (1/(2*lr))*||x_k+1-y_k||^2
            Q = smooth_loss
            for y_k, g_k, x_new in zip(momentum_points, grads, new_x):
                diff = x_new - y_k
                Q += torch.sum(diff * g_k)
                Q += (1 / (2 * current_lr)) * torch.sum(diff * diff)

            if new_smooth <= Q:
                line_search_success = True
                break

            current_lr *= lr_decay

        # Track line search failure for convergence checking
        group['line_search_failed'] = not line_search_success

        # If line search failed completely, use the last computed step
        if new_x is None:
            new_x = []
            for y_k, g_k in zip(momentum_points, grads):
                v = y_k - current_lr * g_k
                new_x.append(penalty.proximal(parameter=v, lr=current_lr, lambda_=lambda_))
            self._set_parameters(params, new_x)

        # Update momentum for next iteration
        # FISTA momentum update: t_k+1 = (1 + sqrt(1 + 4*t_k^2)) / 2
        # y_k+1 = x_k+1 + ((t_k - 1) / t_k+1) * (x_k+1 - x_k)
        for i, p in enumerate(params):
            param_state = self.state[p]
            t_k = param_state['t']

            # Update momentum coefficient
            t_k_plus_1 = (1 + np.sqrt(1 + 4 * t_k * t_k)) / 2
            param_state['t'] = t_k_plus_1

            # Update momentum buffer: y_k+1 = x_k+1 + ((t_k - 1) / t_k+1) * (x_k+1 - x_k)
            x_k_plus_1 = new_x[i]
            x_k_curr = x_k[i]
            momentum_coeff = (t_k - 1) / t_k_plus_1
            param_state['momentum_buffer'] = x_k_plus_1 + momentum_coeff * (x_k_plus_1 - x_k_curr)

        group['lr'] = current_lr
        group['step_count'] += 1

    def get_lr(self):
        """Return current learning rates for each parameter group."""
        return [g['lr'] for g in self.param_groups]

    def line_search_failed(self) -> bool:
        """Check if the line search failed in the last step.

        This can be used as a convergence criterion for FISTA, especially
        when dealing with complex penalties like MCP where the loss might
        oscillate between FISTA and non-penalized parameter updates.

        Returns
        -------
        bool
            True if the line search failed to find a valid step size.
        """
        return any(group.get('line_search_failed', False) for group in self.param_groups)

    def __repr__(self):
        return f"FISTAOptimizer(lr={self.defaults['lr']}, lr_decay={self.defaults['lr_decay']})"

"""
FISTA optimizer implementation for HarderLASSO models.

This module provides the FISTA (Fast Iterative Shrinkage-Thresholding Algorithm)
optimizer for solving sparse optimization problems.
"""

import torch
from torch.optim.optimizer import Optimizer
from math import sqrt
from typing import Callable, Optional, Any, List, Dict


class _FISTAOptimizer(Optimizer):
    """
    FISTA optimizer with backtracking line search and monotone restart.

    This optimizer implements the Fast Iterative Shrinkage-Thresholding Algorithm
    for solving optimization problems of the form:

        minimize F(x) = f(x) + g(x)

    where f(x) is smooth (gradient available) and g(x) is handled by a proximal operator.

    Parameters
    ----------
    params : iterable
        Iterable of parameters to optimize or dicts defining parameter groups.
    proximal_operator : object
        Proximal operator for handling the non-smooth term g(x).
    lr : float, default=0.1
        Learning rate (step size).
    lr_decay : float, default=0.5
        Factor for learning rate decay during line search.
    min_lr : float, default=1e-7
        Minimum learning rate.

    Notes
    -----
    The closure function should accept a backward parameter and return
    both the total loss F(x) and the smooth part f(x).

    Examples
    --------
    >>> from training.proximal_operator import prox_op_factory
    >>> prox_op = prox_op_factory(penalty='lasso', lambda_=0.1)
    >>> optimizer = _FISTAOptimizer(
    ...     model.parameters(),
    ...     proximal_operator=prox_op,
    ...     lr=0.1
    ... )
    """

    def __init__(
        self,
        params,
        proximal_operator=None,
        lr: float = 0.1,
        lr_decay: float = 0.5,
        min_lr: float = 1e-7
    ):
        """Initialize the FISTA optimizer."""
        if lr <= 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if lr_decay <= 0.0 or lr_decay >= 1.0:
            raise ValueError(f"Invalid learning rate decay: {lr_decay}")
        if min_lr <= 0.0:
            raise ValueError(f"Invalid minimum learning rate: {min_lr}")
        if proximal_operator is None:
            raise ValueError("Proximal operator is required for FISTA")

        defaults = dict(
            proximal_operator=proximal_operator,
            lr=lr,
            lr_decay=lr_decay,
            min_lr=min_lr
        )
        super().__init__(params, defaults)

        # Initialize algorithm state
        for group in self.param_groups:
            group['t'] = 1.0
            group['prev_x'] = [p.data.clone() for p in group['params']]
            group['y'] = [p.data.clone() for p in group['params']]
            group['best_loss'] = float('inf')
            group['step_count'] = 0

    def step(self, closure: Callable) -> torch.Tensor:
        """
        Perform a single optimization step.

        Parameters
        ----------
        closure : callable
            A closure that reevaluates the model and returns the loss.
            Must accept backward=True/False parameter.

        Returns
        -------
        torch.Tensor
            The current loss value.
        """
        if closure is None:
            raise ValueError("FISTA requires a closure that returns loss and calls backward().")

        # Perform optimization step for each parameter group
        for group in self.param_groups:
            self._step_group(group, closure)

        # Return current loss value
        return closure(backward=False)

    @torch.no_grad()
    def _set_parameters(self, params: List[torch.Tensor], values: List[torch.Tensor]) -> None:
        """Set parameter values without gradient tracking."""
        for param, value in zip(params, values):
            param.data.copy_(value)

    def _step_group(self, group: Dict[str, Any], closure: Callable) -> None:
        """
        Perform optimization step for a single parameter group.

        Parameters
        ----------
        group : dict
            Parameter group dictionary.
        closure : callable
            Closure function for loss evaluation.
        """
        params = group['params']
        if not params:
            return

        # Get optimizer settings
        lr = group['lr']
        lr_decay = group['lr_decay']
        min_lr = group['min_lr']
        prox_op = group['proximal_operator']

        # Set parameters to current y values
        self._set_parameters(params, group['y'])

        # Evaluate function and gradient at y
        total_loss, smooth_loss = closure(backward=True)
        gradients = [p.grad.clone() for p in params]

        # Backtracking line search
        current_lr = lr
        while True:
            # Compute proximal gradient step
            new_x = []
            for y_k, grad_k in zip(group['y'], gradients):
                v = y_k - current_lr * grad_k
                new_x.append(prox_op.apply(v, current_lr))

            # Evaluate function at new point
            self._set_parameters(params, new_x)
            new_total_loss, new_smooth_loss = closure(backward=False)

            # Check sufficient decrease condition
            with torch.no_grad():
                Q = smooth_loss
                for y_k, grad_k, x_k in zip(group['y'], gradients, new_x):
                    diff = x_k - y_k
                    Q += torch.sum(diff * grad_k)
                    Q += (1 / (2 * current_lr)) * torch.sum(diff * diff)

            if new_smooth_loss <= Q or current_lr <= min_lr:
                break

            current_lr *= lr_decay

        group['lr'] = current_lr

        # Monotone restart check
        if new_total_loss > group['best_loss']:
            group['t'] = 1.0
        else:
            group['best_loss'] = new_total_loss

        # Update FISTA momentum parameter
        t = group['t']
        new_t = (1 + sqrt(1 + 4 * t * t)) / 2
        momentum_coeff = (t - 1) / new_t

        new_y = []
        for x_k, prev_x_k in zip(new_x, group['prev_x']):
            y_k = x_k + momentum_coeff * (x_k - prev_x_k)
            new_y.append(y_k)

        # Update algorithm state
        group['y'] = new_y
        group['prev_x'] = [x.clone() for x in new_x]
        group['t'] = new_t
        group['step_count'] += 1

    def get_lr(self) -> List[float]:
        """
        Get current learning rates for all parameter groups.

        Returns
        -------
        list of float
            Learning rates for each parameter group.
        """
        return [group['lr'] for group in self.param_groups]

    def reset(self) -> None:
        """Reset the optimizer state."""
        for group in self.param_groups:
            group['t'] = 1.0
            group['prev_x'] = [p.data.clone() for p in group['params']]
            group['y'] = [p.data.clone() for p in group['params']]
            group['best_loss'] = float('inf')
            group['step_count'] = 0

    def state_dict(self) -> Dict[str, Any]:
        """
        Return the state of the optimizer as a dictionary.

        Returns
        -------
        dict
            State dictionary containing optimizer state.
        """
        state = super().state_dict()

        # Add FISTA-specific state
        for i, group in enumerate(self.param_groups):
            state['param_groups'][i]['t'] = group['t']
            state['param_groups'][i]['best_loss'] = group['best_loss']
            state['param_groups'][i]['step_count'] = group['step_count']

        return state

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        """
        Load the optimizer state from a dictionary.

        Parameters
        ----------
        state_dict : dict
            State dictionary to load.
        """
        super().load_state_dict(state_dict)

        # Restore FISTA-specific state
        for i, group in enumerate(self.param_groups):
            if 't' in state_dict['param_groups'][i]:
                group['t'] = state_dict['param_groups'][i]['t']
            if 'best_loss' in state_dict['param_groups'][i]:
                group['best_loss'] = state_dict['param_groups'][i]['best_loss']
            if 'step_count' in state_dict['param_groups'][i]:
                group['step_count'] = state_dict['param_groups'][i]['step_count']

    def __repr__(self) -> str:
        """String representation of the optimizer."""
        return f"_FISTAOptimizer(lr={self.defaults['lr']}, lr_decay={self.defaults['lr_decay']})"

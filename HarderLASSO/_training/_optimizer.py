import torch
import numpy as np
from torch.optim.optimizer import Optimizer
from typing import Callable, List, Any, Dict, Tuple

class _IdentityPenalty:
      """Identity proximal operator - no regularization."""
      def proximal(self, parameter, lr, lambda_):
          return parameter

class _ISTAOptimizer(Optimizer):
      """
      ISTA optimizer with backtracking line search.

      Solves:
          minimize F(x) = f(x) + g(x)
      where f is smooth and g is handled via a proximal operator.

      Supports multiple parameter groups with different proximal operators.
      If no penalty is provided for a group, uses identity (standard gradient descent).
      """

      def __init__(
          self,
          params,
          penalty=None,
          lr: float = 0.1,
          lambda_: float = 0.0,
          lr_decay: float = 0.5,
          min_lr: float = 1e-7
      ):
          if lr <= 0.0:
              raise ValueError(f"Invalid learning rate: {lr}")
          if not (0.0 < lr_decay < 1.0):
              raise ValueError(f"Invalid learning rate decay: {lr_decay}")
          if min_lr <= 0.0:
              raise ValueError(f"Invalid minimum learning rate: {min_lr}")

          # Default penalty (identity) if none provided
          if penalty is None:
              penalty = _IdentityPenalty()

          defaults = dict(
              penalty=penalty,
              lr=lr,
              lambda_=lambda_,
              lr_decay=lr_decay,
              min_lr=min_lr
          )
          super().__init__(params, defaults)

          # Initialize each group and set default penalty if needed
          for group in self.param_groups:
              if 'penalty' not in group or group['penalty'] is None:
                  group['penalty'] = _IdentityPenalty()
              elif not hasattr(group['penalty'], 'proximal'):
                  raise ValueError("Penalty function must implement a 'proximal' method")

              group.setdefault("lambda_", 0.0)

      @torch.no_grad()
      def _set_parameters(self, params: List[torch.Tensor], values: List[torch.Tensor]):
          """Set parameter values efficiently."""
          for p, v in zip(params, values):
              p.data.copy_(v)

      def step(self, closure: Callable[[], Tuple[torch.Tensor, torch.Tensor]]) -> torch.Tensor:
          """
          Perform one ISTA step.

          closure should:
          - When called with backward=True: compute gradients and return (total_loss, smooth_loss)
          - When called with backward=False: return (total_loss, _)

          Returns the final total loss.
          """
          if closure is None:
              raise ValueError("ISTA requires a closure that returns loss and calls backward()")

          # Process each parameter group independently
          for group in self.param_groups:
              self._step_group(group, closure)
          return closure(backward=False)

      def _step_group(self, group: Dict[str, Any], closure: Callable) -> None:
          """Perform ISTA step for a single parameter group."""
          params = group["params"]
          lr = group["lr"]
          lr_decay = group["lr_decay"]
          min_lr = group["min_lr"]
          lambda_ = group["lambda_"]
          penalty = group["penalty"]

          # Save current point
          x_old = [p.data.clone() for p in params]

          # Set parameters to current point and compute gradient
          self._set_parameters(params, x_old)
          _, smooth_old = closure(backward=True)

          # Extract gradients (handle case where grad is None)
          grads = []
          for p in params:
              if p.grad is not None:
                  grads.append(p.grad.clone())
              else:
                  grads.append(torch.zeros_like(p))

          # Backtracking line search
          current_lr = lr
          success = False

          while current_lr >= min_lr:
              # Compute proximal-gradient step: x_new = prox_{lr*g}(x_old - lr * grad_f(x_old))
              new_x = []
              for x_k, g_k in zip(x_old, grads):
                  v = x_k - current_lr * g_k
                  x_new = penalty.proximal(parameter=v, lr=current_lr, lambda_=lambda_)
                  new_x.append(x_new)

              # Set parameters to new point and evaluate smooth part
              self._set_parameters(params, new_x)
              _, smooth_new = closure(backward=False)

              # Check Armijo condition for the smooth part only
              # Q_L(x_new, x_old) = f(x_old) + <grad_f(x_old), x_new - x_old> + L/2 * ||x_new - x_old||^2
              armijo_bound = smooth_old
              for x_k, g_k, x_new in zip(x_old, grads, new_x):
                  diff = x_new - x_k
                  # Linear term: <gradient, difference>
                  armijo_bound += torch.sum(g_k * diff)
                  # Quadratic term: (1/2L) * ||difference||^2, where L = 1/lr
                  armijo_bound += (1.0 / (2.0 * current_lr)) * torch.sum(diff * diff)

              # Accept step if smooth part satisfies sufficient decrease
              if smooth_new <= armijo_bound:
                  success = True
                  break

              current_lr *= lr_decay

          if success:
              group["lr"] = current_lr
          else:
              # If line search failed, revert to old point
              self._set_parameters(params, x_old)
              group["lr"] = max(current_lr, min_lr)

      def get_lr(self) -> List[float]:
          """Get current learning rates for all parameter groups."""
          return [g["lr"] for g in self.param_groups]

      def __repr__(self):
          return (f"ISTAOptimizer(lr={self.defaults['lr']}, "
                  f"lr_decay={self.defaults['lr_decay']}, "
                  f"min_lr={self.defaults['min_lr']})")

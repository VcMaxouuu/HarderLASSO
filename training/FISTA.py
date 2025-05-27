import torch
from torch.optim.optimizer import Optimizer
from math import sqrt

class FISTA(Optimizer):
    """
    FISTA optimizer with backtracking line search and monotone restart

    Minimizes F(x) = f(x) + g(x), where f is smooth (gradient via closure)
    and g is handled by the proximal operator.

    Closure should accept backward=True/False, run backward only if True.
    Closure should return the F(x) and f(x).
    """
    def __init__(
        self,
        params,
        proximal_operator=None,
        lr=0.1,
        lr_decay=0.5,
        min_lr=1e-7
    ):
        defaults = dict(
            proximal_operator=proximal_operator, lr=lr, lr_decay=lr_decay, min_lr=min_lr
        )
        super().__init__(params, defaults)
        for group in self.param_groups:
            group['t'] = 1.0
            group['prev_x'] = [p.data.clone() for p in group['params']]
            group['y'] = [p.data.clone() for p in group['params']]
            group['best_loss'] = float('inf')

    def step(self, closure):
        if closure is None:
            raise ValueError("FISTA requires a closure that returns loss and calls backward().")
        for idx, group in enumerate(self.param_groups):
            self._step_group(group, closure)
        return closure(backward=False)

    @torch.no_grad()
    def _set_params(self, params, vals):
        for p, v in zip(params, vals):
            p.data.copy_(v)

    def _step_group(self, group, closure):
        params = group['params']
        if not params:
            return
        lr, lr_decay, lr_min = group['lr'], group['lr_decay'], group['min_lr']
        prox_opp = group['proximal_operator']

        self._set_params(params, group['y'])
        Fy, fy = closure(backward=True)
        grads = [p.grad.clone() for p in params]

        curr_lr = lr
        while True:
            new_x = []
            for y_k, g_k in zip(group['y'], grads):
                v = y_k - curr_lr * g_k
                new_x.append(prox_opp.apply(v, curr_lr))

            self._set_params(params, new_x)
            Fply, fply = closure(backward=False)

            with torch.no_grad():
                Q = fy
                for idx, _ in enumerate(params):
                    diff = new_x[idx] - group['y'][idx]
                    Q += torch.sum(diff * grads[idx])
                    Q += (1/(2*curr_lr)) * torch.sum(diff * diff)

            if fply < Q or curr_lr < lr_min:
                break
            curr_lr *= lr_decay

        group['lr'] = curr_lr

        if Fply > group['best_loss']:
            group['t'] = 1.0
        else:
            group['best_loss'] = Fply

        t = group['t']
        new_t = (1 + sqrt(1 + 4 * t ** 2)) / 2
        coeff = (t - 1) / new_t

        proposed_y = []
        for idx, y_k in enumerate(group['y']):
            y_k = new_x[idx] + coeff * (new_x[idx] - group['prev_x'][idx])
            proposed_y.append(y_k)

        group['y'] = proposed_y
        group['prev_x'] = [x.clone() for x in new_x]
        group['t'] = new_t

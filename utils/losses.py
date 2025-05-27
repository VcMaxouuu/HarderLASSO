import torch.nn as nn
import torch
import math

# -------------------------------------------------
# Pivotal Regression Loss
# -> square root SE
# -------------------------------------------------
class RSELoss(nn.Module):
    """Root Square Error loss strategy."""
    def __init__(self, reduction='mean'):
        super(RSELoss, self).__init__()
        self.crit = nn.MSELoss(reduction=reduction)

    def forward(self, input, target):
        return torch.sqrt(self.crit(input, target))


# -------------------------------------------------
# Classification Loss
# -> cross entropy
# -------------------------------------------------
class ClassificationLoss(nn.Module):
    """Classification loss strategy."""
    def __init__(self, reduction='mean'):
        super(ClassificationLoss, self).__init__()
        self.crit = nn.CrossEntropyLoss(reduction=reduction)

    def forward(self, input, target):
        return self.crit(input, target)


# -------------------------------------------------
# Pivotal Cox Partial Likelihood Loss
# -> square root Partial Likelihood
# -------------------------------------------------
class CoxSquareLoss(nn.Module):
    def __init__(self, reduction='mean', approximation='Breslow'):
        super(CoxSquareLoss, self).__init__()

        if approximation not in {'Efron', 'Breslow', None}:
            raise ValueError(f"Invalid approximation method: {approximation}. Supported options are 'Efron', 'Breslow' or None.")

        self.reduction = reduction
        self.approximation = approximation
        self._initialized = False

    def _initialize(self, target):
        times    = target[:,0]
        sort_idx = torch.argsort(-times)
        self.register_buffer('sort_idx', sort_idx)

        times  = times[sort_idx]
        events = target[:,1][sort_idx]
        device = times.device

        # Unique event times and grouping
        unique_times, inv_idx, counts = torch.unique_consecutive(
            times, return_inverse=True, return_counts=True
        )
        starts = torch.cat([counts.new_zeros(1), counts.cumsum(0)[:-1]])
        ends = counts.cumsum(0)

        # Sum of uncensored events per unique time
        sum_uncens = torch.zeros_like(unique_times, dtype=torch.float, device=device)
        for i in range(len(unique_times)):
            s, e = starts[i].item(), ends[i].item()
            sum_uncens[i] = events[s:e].sum()

        # Register buffers so they move with module and are saved in state_dict
        self.register_buffer('events', events)
        self.register_buffer('starts', starts)
        self.register_buffer('ends', ends)
        self.register_buffer('counts', counts)
        self.register_buffer('unique_times', unique_times)
        self.register_buffer('sum_uncens', sum_uncens)
        self.register_buffer('single_event_mask', sum_uncens == 1)
        self.register_buffer('multi_event_idx', (sum_uncens > 1).nonzero().squeeze(1))

        # Precompute Efron denominators if needed
        if self.approximation == 'Efron':
            max_ties = int(sum_uncens.max().item())
            if max_ties > 1:
                for idx in self.multi_event_idx.tolist():
                    d_i = int(sum_uncens[idx].item())
                    denom = torch.arange(d_i, device=device, dtype=torch.float) / d_i
                    # store each tied denominator as buffer
                    self.register_buffer(f'tied_denom_{idx}', denom)

        self._initialized = True


    def forward(self, input, target):
        input = input.squeeze()

        if not self._initialized:
            self._initialize(target)

        input = input[self.sort_idx]

        # Case 1: No tied events or no approximation
        if self.approximation is None or (self.sum_uncens <= 1).all():
            log_cum = torch.logcumsumexp(input, dim=0)
            loss = - ((input - log_cum) * self.events).sum()

        # Case 2 : Tied events -> Using approximation
        if self.approximation == 'Breslow':
            scaled_log_cumsums = torch.logcumsumexp(input, dim=0)[self.ends - 1] * self.sum_uncens
            loss = - (input * self.events).sum() + scaled_log_cumsums.sum()

        if self.approximation == 'Efron':
            log_cum = torch.logcumsumexp(input, dim=0)
            single_idx = (self.sum_uncens == 1).nonzero().squeeze(1)
            loss_term = log_cum[self.ends - 1][single_idx].sum()

            # Terms for ties
            for idx in self.multi_event_idx.tolist():
                s, e = self.starts[idx].item(), self.ends[idx].item()
                d_i = int(self.sum_uncens[idx].item())
                # risk set is [0:e]
                local_max = input[:e].max()
                risk_sum = torch.exp(input[:e] - local_max).sum()
                tied_exp = torch.exp(input[s:e][self.events[s:e].bool()] - local_max).sum()
                denom = getattr(self, f'tied_denom_{idx}')
                for k in range(d_i):
                    efron = risk_sum - denom[k] * tied_exp
                    loss_term += torch.log(efron) + local_max

            loss = - (input * self.events).sum() + loss_term

        if self.reduction == 'mean':
            loss = loss / input.numel()
        return torch.sqrt(loss)

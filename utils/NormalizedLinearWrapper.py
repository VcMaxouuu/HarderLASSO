
import torch
import torch.nn as nn
import torch.nn.functional as F


class NormalizedLinearWrapper(nn.Module):
    def __init__(self, linear: nn.Linear):
        super().__init__()
        self.linear = linear

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Normalize the weight along its columns (dim=1)
        normed_weight = F.normalize(self.linear.weight, p=2, dim=1)
        return F.linear(x, normed_weight, self.linear.bias)

    @property
    def weight(self):
        return self.linear.weight

    @property
    def bias(self):
        return self.linear.bias

    @property
    def in_features(self):
        return self.linear.in_features

    @property
    def out_features(self):
        return self.linear.out_features

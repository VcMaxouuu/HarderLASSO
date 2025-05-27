from .qut_mixin import QUTMixin
import torch

class RegressionQUT(QUTMixin):
    """
    QUT manager for regression tasks.
    """
    def __init__(self, lambda_qut=None, n_samples=5000, alpha=0.05):
        super().__init__(lambda_qut, n_samples, alpha)

    def _simulation_method(self, X, target, n_reps):
        device, dtype = X.device, X.dtype
        n_samples, _ = X.shape

        y_sample = torch.randn(n_samples, n_reps, dtype=dtype, device=device)
        y_centered = y_sample - y_sample.mean(dim=0, keepdim=True)

        Xy = torch.matmul(X.T, y_centered)
        Xy_inf_norm = torch.abs(Xy).amax(dim=0)
        y_norm = y_centered.norm(p=2, dim=0)
        return Xy_inf_norm / y_norm


class ClassificationQUT(QUTMixin):
    """
    QUT manager for classification tasks.
    """
    def __init__(self, lambda_qut=None, n_samples=5000, alpha=0.05):
        super().__init__(lambda_qut, n_samples, alpha)

    def _compute_class_distribution(self, target):
        """
        Computes the empirical class distribution p for sampling.
        This uses PyTorch's bincount on the integer targets.
        """
        num_classes = target.unique().numel()
        counts = torch.bincount(target, minlength=num_classes).float()
        return counts / target.size(0)

    def _simulation_method(self, X, target, n_reps):
        device, dtype = X.device, X.dtype
        hat_p = self._compute_class_distribution(target).to(device=device, dtype=dtype)

        y_sample_indices = torch.multinomial(hat_p, X.shape[0] * n_reps, replacement=True)
        y_sample_indices = y_sample_indices.reshape(n_reps, X.shape[0])

        num_classes = len(hat_p)
        y_sample_one_hot = torch.nn.functional.one_hot(y_sample_indices, num_classes=num_classes).float()
        y_centered = y_sample_one_hot - y_sample_one_hot.mean(dim=1, keepdim=True)

        xy = torch.einsum('ij,rjk->rik', X.T, y_centered)
        xy_sum = torch.abs(xy).sum(dim=2)
        l = xy_sum.max(dim=1)[0]
        return l


class CoxQUT(QUTMixin):
    """
    QUT manager for Cox regression tasks.
    """
    def __init__(self, lambda_qut=None, n_samples=5000, alpha=0.05):
        super().__init__(lambda_qut, n_samples, alpha)

    def _simulation_method(self, X, target, n_reps):
        device, dtype = X.device, X.dtype
        n_samples, _ = X.shape
        times = target[:, 0]
        events = target[:, 1]

        pair_boot_idx = torch.randint(0, n_samples, (n_samples, n_reps), dtype=torch.long, device=device)
        times_boot = times[pair_boot_idx]
        events_boot = events[pair_boot_idx]     # (n_samples, n_reps)

        counts = torch.arange(1, n_samples+1, dtype=dtype, device=device).unsqueeze(1) # (n_samples, 1)
        cumsums_mean = X.cumsum(0).div(counts)  # (n_samples, n_features)
        X_diff = X - cumsums_mean               # (n_samples, n_features)

        grad_nom = -(X_diff.unsqueeze(1) * events_boot.unsqueeze(2)).sum(dim=0) # (n_reps, n_features)
        grad_denom = (events_boot * torch.log(counts.squeeze(1)).unsqueeze(1)).sum(0).sqrt().mul(2) # (n_reps, )
        quotient = grad_nom/(grad_denom + 1e-10).unsqueeze(1) # (n_reps, n_features)
        return (quotient.abs().amax(1))

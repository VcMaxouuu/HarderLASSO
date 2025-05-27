import torch
from numpy import ndarray, asarray, float32
from sklearn.preprocessing import StandardScaler
from .task_mixin import TaskMixin
from utils.losses import RSELoss

class RegressorTaskMixin(TaskMixin):
    """
    A mixin class for regression tasks, extending the `TaskMixin` interface.

    This class provides implementations for preprocessing data, initializing penalized parameters,
    defining a regression-specific loss criterion, scoring predictions, and formatting outputs.
    """

    def __init__(self):
        """
        Initializes the RegressorTaskMixin.

        Sets up the scaler for preprocessing input features.
        """
        super().__init__()
        self.scaler = None

    def preprocess_data(self,
                   X: ndarray,
                   target: ndarray | None = None,
                   fit_scaler: bool = True) -> tuple[torch.Tensor, torch.Tensor | None]:

        if self.scaler is None and fit_scaler:
            self.scaler = StandardScaler()

        X = asarray(X, dtype=float32)
        if X.ndim == 1:
            X = X.reshape(-1, 1)

        if fit_scaler:
            X_tensor = torch.tensor(self.scaler.fit_transform(X), dtype=torch.float)
        else:
            X_tensor = torch.tensor(self.scaler.transform(X), dtype=torch.float)

        if target is None:
            return X_tensor

        target = asarray(target, dtype= float32)
        if target.ndim == 2:
            if target.shape[1] == 1:
                target = target.squeeze(axis=1)
            else:
                raise ValueError("Target should be of shape (n, ) or (n, 1). Got: {}".format(target.shape))


        target_tensor = torch.tensor(target, dtype=torch.float)
        return X_tensor, target_tensor

    def criterion(self, outputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        return RSELoss(reduction='sum')(outputs, targets)

    def score(self, X: ndarray, target: ndarray) -> dict[str, float]:
        from numpy import mean, abs, sum
        target = asarray(target, dtype=float32)

        if target.ndim == 2:
            if target.shape[1] == 1:
                target = target.reshape(-1)
            else:
                raise ValueError("Target should be of shape (n, ) or (n, 1). Got: {}".format(target.shape))

        pred = self.predict(X)
        mse = mean((target - pred) ** 2)
        mae = mean(abs(target - pred))
        target_variance = sum((target - mean(target)) ** 2)
        if target_variance == 0:
            r2 = float('nan')
        else:
            r2 = 1 - (sum((target - pred) ** 2) / target_variance)

        return {
            "MSE": mse,
            "MAE": mae,
            "R2": r2
        }

    def format_predictions(self, raw_outputs: ndarray) -> ndarray:
        return raw_outputs.squeeze().cpu().numpy()

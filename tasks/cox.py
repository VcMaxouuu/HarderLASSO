import torch
import numpy as np
from sklearn.preprocessing import StandardScaler
from .task_mixin import TaskMixin
from utils.losses import CoxSquareLoss

class CoxTaskMixin(TaskMixin):
    """
    A mixin class for Cox regression tasks, extending the `TaskMixin` interface.

    This class provides implementations for preprocessing data,
    defining a cox-specific loss criterion, scoring predictions, and formatting outputs.
    """
    def __init__(self, approximation: str = 'Breslow'):
        super().__init__()
        self.scaler = None
        self.approximation = approximation
        self._loss = CoxSquareLoss(reduction='sum', approximation=approximation)

    def preprocess_data(self,
                   X: np.ndarray,
                   target: np.ndarray | None = None,
                   fit_scaler: bool = True) -> tuple[torch.Tensor, torch.Tensor | None]:

        if self.scaler is None and fit_scaler:
            self.scaler = StandardScaler()

        X = np.asarray(X, dtype=np.float32)
        if X.ndim == 1:
            X = X.reshape(-1, 1)

        if fit_scaler:
            X_tensor = torch.tensor(self.scaler.fit_transform(X), dtype=torch.float)
        else:
            X_tensor = torch.tensor(self.scaler.transform(X), dtype=torch.float)

        if target is None:
            return X_tensor

        target = np.asarray(target, dtype=np.float32)
        if target.ndim != 2:
            raise ValueError("Target should be a 2D array or a tuple of two arrays (durations, events).")
        if target.shape[0] == 2 and target.shape[1] != 2:
            target = target.T
        if target.shape[1] != 2:
            raise ValueError("Target must have exactly two columns: (durations, events).")

        target_tensor = torch.from_numpy(target)
        return X_tensor, target_tensor

    def criterion(self, outputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        return self._loss(outputs, targets)

    def score(self, X: np.ndarray, target: np.ndarray) -> dict[str, float]:
        from utils.cox_utils import concordance_index, cox_partial_log_likelihood

        target = np.asarray(target, dtype=np.float32)

        if target.ndim != 2:
            raise ValueError("Target should be a 2D array or a tuple of two arrays (durations, events).")
        else:
            if target.shape[0] == 2 and target.shape[1] != 2:
                target = target.T

        times = target[:, 0]
        events = target[:, 1].astype(int)
        pred = self.predict(X)

        c_index = concordance_index(times, events, pred)
        npll = cox_partial_log_likelihood(times, events, pred)

        return {"C-index": c_index, "neg_partial_log_likelihood": npll}


    def format_predictions(self, raw_outputs: np.ndarray) -> np.ndarray:
        return raw_outputs.squeeze().cpu().numpy()

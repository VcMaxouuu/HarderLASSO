import torch
from numpy import ndarray, asarray, float32
from sklearn.preprocessing import StandardScaler
from .task_mixin import TaskMixin
from utils.losses import ClassificationLoss
from numpy import argmax, unique

class ClassifierTaskMixin(TaskMixin):
    """
    A mixin class for classification tasks, extending the `TaskMixin` interface.

    This class provides implementations for preprocessing data, initializing penalized parameters,
    defining a classification-specific loss criterion, scoring predictions, and formatting outputs.
    """

    def __init__(self):
        """
        Initializes the ClassifierTaskMixin.

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

        self.NN._output_dim = len(unique(target))

        target_tensor = torch.tensor(target, dtype=torch.long)
        return X_tensor, target_tensor

    def criterion(self, outputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        return ClassificationLoss(reduction='sum')(outputs, targets)

    def score(self, X: ndarray, target: ndarray) -> dict[str, float]:
        target = asarray(target, dtype=float32)

        if target.ndim == 2:
            if target.shape[1] == 1:
                target = target.reshape(-1)
            else:
                raise ValueError("Target should be of shape (n, ) or (n, 1). Got: {}".format(target.shape))

        pred = self.predict(X)
        accuracy = (pred == target).mean()
        return {
            "accuracy": accuracy
        }

    def format_predictions(self, raw_outputs: ndarray) -> ndarray:
        outputs = raw_outputs.squeeze().cpu().numpy()
        return argmax(outputs, axis=1)

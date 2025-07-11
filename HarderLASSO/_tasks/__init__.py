"""
Task-specific implementations for the HarderLASSO framework.

This module provides task mixins for different machine learning problems,
each implementing the TaskMixin interface with task-specific preprocessing,
loss functions, and evaluation metrics.
"""

from ._base import _BaseTaskMixin
from ._classification import _ClassificationTaskMixin
from ._cox import _CoxTaskMixin
from ._regression import _RegressionTaskMixin

__all__ = [
    "_BaseTaskMixin",
    "_RegressionTaskMixin",
    "_ClassificationTaskMixin"
    "_CoxTaskMixin"
]

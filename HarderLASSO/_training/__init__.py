"""
Private training utilities for HarderLASSO models.

This module contains internal training components including optimizers,
trainers, and callbacks.
"""

from ._trainer import _FeatureSelectionTrainer
from ._optimizer import _FISTAOptimizer
from ._callbacks import _ConvergenceChecker, _LoggingCallback
from ._proximal import _create_proximal_operator

__all__ = [
    '_FeatureSelectionTrainer',
    '_FISTAOptimizer',
    '_ConvergenceChecker',
    '_LoggingCallback',
    '_create_proximal_operator'
]

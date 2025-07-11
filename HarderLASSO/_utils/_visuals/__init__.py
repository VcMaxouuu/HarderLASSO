"""Internal visualization utilities for neural network feature selection.

This package contains visualization functions for different types of models
and results. These are not part of the public API.
"""
import os
import matplotlib.pyplot as plt

_style_path = os.path.join(os.path.dirname(__file__), "visuals_style.mplstyle")
plt.style.use(_style_path)

from ._cox_plots import (
    _plot_baseline_cumulative_hazard,
    _plot_baseline_survival_function,
    _plot_survival_curves,
    _plot_feature_effects,
    _plot_kaplan_meier,
)
from ._network_plots import plot_lollipop
from ._regression_plots import (
    _plot_actual_vs_predicted,
    _plot_residuals_vs_predicted,
    _plot_residual_distribution,
    _plot_qq,
)

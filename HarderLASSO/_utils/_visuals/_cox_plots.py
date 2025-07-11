"""Cox regression visualization utilities.

This module provides plotting functions for Cox regression results,
including baseline hazard, survival curves, and feature effects.
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from typing import Optional, List, Union, Callable
from numpy import ndarray


def _plot_baseline_cumulative_hazard(
    baseline_hazard_df: pd.DataFrame,
    ax: Optional[plt.Axes] = None,
    **plot_kwargs
) -> plt.Axes:
    """
    Plot the estimated baseline cumulative hazard function.

    Parameters
    ----------
    baseline_hazard_df : pd.DataFrame
        DataFrame containing the baseline cumulative hazard with time values
        as index and hazard values as column.
    ax : matplotlib.axes.Axes, optional
        Matplotlib axes object. If None, creates new figure.
    **plot_kwargs
        Additional keyword arguments passed to pandas.DataFrame.plot.

    Returns
    -------
    matplotlib.axes.Axes
        Matplotlib axes object containing the plot.
    """
    if ax is None:
        fig, ax = plt.subplots()

    baseline_hazard_df.plot(ax=ax, **plot_kwargs)
    ax.set_title("Baseline Cumulative Hazard")
    ax.set_xlabel("Time")
    ax.set_ylabel(r"$H_0(t)$")

    return ax


def _plot_baseline_survival_function(
    baseline_survival_df: pd.DataFrame,
    ax: Optional[plt.Axes] = None,
    **plot_kwargs
) -> plt.Axes:
    """
    Plot the estimated baseline survival function.

    Parameters
    ----------
    baseline_survival_df : pd.DataFrame
        DataFrame containing the baseline survival with time values
        as index and survival values as column.
    ax : matplotlib.axes.Axes, optional
        Matplotlib axes object. If None, creates new figure.
    **plot_kwargs
        Additional keyword arguments passed to pandas.DataFrame.plot.

    Returns
    -------
    matplotlib.axes.Axes
        Matplotlib axes object containing the plot.
    """
    if ax is None:
        fig, ax = plt.subplots()

    baseline_survival_df.plot(ax=ax, **plot_kwargs)
    ax.set_title("Baseline Survival Function")
    ax.set_xlabel("Time")
    ax.set_ylabel(r"$S_0(t)$")

    return ax


def _plot_survival_curves(
    durations: ndarray,
    survival_matrix: ndarray,
    labels: List[str],
    individuals_idx: Optional[Union[int, List[int]]] = None,
    ax: Optional[plt.Axes] = None,
    **plot_kwargs
) -> plt.Axes:
    """
    Plot survival curves for multiple individuals.

    Parameters
    ----------
    durations : ndarray
        Time points of shape (n_times,).
    survival_matrix : ndarray
        Survival probabilities of shape (n_times, n_individuals) where each
        column represents an individual's survival over time.
    labels : list of str
        Names for each individual (length == n_individuals).
    individuals_idx : int or list of int, optional
        Which individuals to plot. If None, plots all. Can be a single
        integer or list of integers.
    ax : matplotlib.axes.Axes, optional
        Matplotlib axes object. If None, creates new figure.
    **plot_kwargs
        Additional keyword arguments passed to matplotlib.pyplot.plot.

    Returns
    -------
    matplotlib.axes.Axes
        Matplotlib axes object containing the plot.

    Raises
    ------
    ValueError
        If individuals_idx contains out-of-bounds indices.
    """
    n_ind = survival_matrix.shape[1]

    if individuals_idx is None:
        inds = list(range(n_ind))
    elif isinstance(individuals_idx, int):
        inds = [individuals_idx]
    else:
        inds = individuals_idx

    if ax is None:
        fig, ax = plt.subplots()

    for idx in inds:
        if 0 <= idx < n_ind:
            ax.plot(
                durations,
                survival_matrix[:, idx],
                label=labels[idx],
                **plot_kwargs
            )
        else:
            raise ValueError(
                f"Index {idx} is out of bounds for individuals (0 to {n_ind-1})."
            )

    ax.set_title("Survival Curves")
    ax.set_xlabel("Time")
    ax.set_ylabel("Survival Probability")
    ax.legend()

    return ax


def _plot_feature_effects(
    durations: ndarray,
    baseline_survival: ndarray,
    X: ndarray,
    predict_fn: Callable,
    feature_name: str,
    idx: int,
    values: Optional[List] = None,
    ax: Optional[plt.Axes] = None,
    **plot_kwargs
) -> plt.Axes:
    """
    Plot the effect of a single feature on survival curves.

    This function shows how varying a specific feature affects survival
    while holding all other features at their mean values.

    Parameters
    ----------
    durations : ndarray
        Time points at which survival is evaluated of shape (n_times,).
    baseline_survival : ndarray
        Baseline survival curve S0(t) of shape (n_times,).
    X : ndarray
        Original feature matrix of shape (n_samples, n_features) used to
        compute means and create synthetic data.
    predict_fn : callable
        Function mapping feature matrix to linear predictors (log-hazards).
    feature_name : str
        Name of the feature to vary.
    idx : int
        Index of the feature in the feature matrix.
    values : list, optional
        Specific values of the feature to plot. If None, uses all unique
        values when there are ≤5, otherwise uses quantiles.
    ax : matplotlib.axes.Axes, optional
        Matplotlib axes object. If None, creates new figure.
    **plot_kwargs
        Additional keyword arguments passed to matplotlib.pyplot.plot.

    Returns
    -------
    matplotlib.axes.Axes
        Matplotlib axes object containing the plot.

    Notes
    -----
    The function creates synthetic data by setting all features to their
    mean values except for the feature of interest, which is varied
    according to the specified values.
    """
    X = np.asarray(X)

    col = X[:, idx]
    unique_vals = np.unique(col)

    if values is None:
        if unique_vals.size <= 5:
            values = unique_vals.tolist()
        else:
            # Use quantiles for continuous features
            qs = np.linspace(0, 1, 5)
            values = np.quantile(col, qs).tolist()

    # Build a "mean" template for X
    mean_vec = X.mean(axis=0, keepdims=True)  # shape (1, n_features)
    n = X.shape[0]

    if ax is None:
        fig, ax = plt.subplots()

    for v in values:
        # Create synthetic data with feature set to specific value
        Xv = np.tile(mean_vec, (n, 1))
        Xv[:, idx] = v

        # Compute survival curves: S0(t) ** exp(linear_predictor)
        lp = predict_fn(Xv)
        surv = baseline_survival[:, None] ** np.exp(lp)
        curve = surv.mean(axis=1)

        ax.plot(durations, curve, label=f"{round(v, 3)}", **plot_kwargs)

    ax.set_title(f"Survival Curves by {feature_name}")
    ax.set_xlabel("Time")
    ax.set_ylabel("Survival Probability")
    ax.legend(title='Value')

    return ax


def _plot_kaplan_meier(
    times: ndarray,
    events: ndarray,
    groups: Optional[ndarray] = None,
    ax: Optional[plt.Axes] = None,
    **plot_kwargs
) -> plt.Axes:
    """
    Plot Kaplan-Meier survival curves.

    Parameters
    ----------
    times : ndarray
        Survival times of shape (n_samples,).
    events : ndarray
        Event indicators of shape (n_samples,) where 1=event, 0=censored.
    groups : ndarray, optional
        Group labels for stratified analysis.
    ax : matplotlib.axes.Axes, optional
        Matplotlib axes object. If None, creates new figure.
    **plot_kwargs
        Additional keyword arguments passed to matplotlib.pyplot.plot.

    Returns
    -------
    matplotlib.axes.Axes
        Matplotlib axes object containing the plot.
    """
    if ax is None:
        fig, ax = plt.subplots()

    if groups is None:
        groups = np.zeros(len(times))

    for group_val in np.unique(groups):
        group_mask = groups == group_val
        group_times = times[group_mask]
        group_events = events[group_mask]

        # Compute Kaplan-Meier estimate
        unique_times = np.unique(group_times)
        survival_prob = np.ones(len(unique_times))

        for i, t in enumerate(unique_times):
            at_risk = (group_times >= t).sum()
            events_at_t = ((group_times == t) & (group_events == 1)).sum()

            if at_risk > 0:
                survival_prob[i] = 1 - events_at_t / at_risk

        # Compute cumulative survival
        survival_prob = np.cumprod(survival_prob)

        ax.step(unique_times, survival_prob, where='post',
                label=f'Group {group_val}', **plot_kwargs)

    ax.set_title("Kaplan-Meier Survival Curves")
    ax.set_xlabel("Time")
    ax.set_ylabel("Survival Probability")
    ax.set_ylim(0, 1)
    if len(np.unique(groups)) > 1:
        ax.legend()

    return ax

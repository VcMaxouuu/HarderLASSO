import numpy as np
import matplotlib.pyplot as plt

def plot_baseline_cumulative_hazard(baseline_hazard_df, ax=None, **plot_kwargs):
    """
    Plots the estimated baseline cumulative hazard.

    Args:
        baseline_hazard_df (DataFrame): Dataframe containing the baseline cumulative hazard and time values.
        ax : matplotlib.axes.Axes, optional
            If provided, the plot will be drawn on this axes.
    """
    if ax is None:
        fig, ax = plt.subplots()
    baseline_hazard_df.plot(ax=ax, **plot_kwargs)
    ax.set_title("Baseline Cumulative Hazard")
    ax.set_xlabel("Time")
    ax.set_ylabel(r"$H_0(t)$")
    return ax

def plot_baseline_survival_function(baseline_survival_df, ax=None, **plot_kwargs):
    """
    Plots the estimated baseline survival function.

    Args:
        baseline_survival_df (DataFrame): Dataframe containing the baseline survival and time values.
        ax : matplotlib.axes.Axes, optional
            If provided, the plot will be drawn on this axes.
    """
    if ax is None:
        fig, ax = plt.subplots()
    baseline_survival_df.plot(ax=ax, **plot_kwargs)
    ax.set_title("Baseline Survival")
    ax.set_xlabel("Time")
    ax.set_ylabel(r"$S_0(t)$")
    return ax

def plot_survival_curves(
    durations: np.ndarray,
    survival_matrix: np.ndarray,
    labels: list[str],
    individuals_idx: int | list[int] | None = None,
    ax=None,
    **plot_kwargs
):
    """
    Plots survival curves given raw arrays.

    Args:
        durations : array of shape (n_times,)
            The time points.
        survival_matrix : array of shape (n_times, n_individuals)
            Each column is an individual's survival over time.
        labels : list of str
            Names for each individual (length == n_individuals).
        individuals_idx : int or list of ints, optional
            Which individuals to plot. Default is all.
        figsize : tuple, optional
            Size of the matplotlib figure.
        ax : matplotlib.axes.Axes, optional
            If provided, the plot will be drawn on this axes.
        **plot_kwargs :
            Additional kwargs passed to plt.plot.
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
            raise ValueError(f"Index {idx} is out of bounds for individuals (0 to {n_ind-1}).")

    ax.set_title("Survival Curves")
    ax.set_xlabel("Time")
    ax.set_ylabel("Survival Probability")
    ax.legend()
    return ax


def plot_feature_effects(
    durations: np.ndarray,
    baseline_survival: np.ndarray,
    X: np.ndarray,
    predict_fn: callable,
    feature_name: list[str],
    idx: int,
    values: list | None = None,
    ax=None,
    **plot_kwargs
):
    """
    Plot the effect of a single feature on survival, holding all other features
    at their mean.

    Parameters
    ----------
    durations : array-like, shape (n_times,)
        The time points at which survival is evaluated.
    baseline_survival : array-like, shape (n_times,)
        The baseline survival curve S0(t).
    X : array-like, shape (n_samples, n_features)
        Original feature matrix (used to compute means and to broadcast).
    predict_fn : callable
        Function mapping feature matrix -> linear predictors (log-hazards).
    feature_name : str
        The feature to vary.
    idx : int
        The index of the feature in the feature matrix.
    values : list of scalars, optional
        Which values of `feature_name` to plot.
        If None, uses all unique values when there are ≤5,
        otherwise uses the [0%,25%,50%,75%,100%] quantiles.
    ax : matplotlib.axes.Axes, optional
        If provided, the plot will be drawn on this axes.
    **plot_kwargs :
        Passed to `ax.plot()` (e.g. `linestyle`, `color`).
    """
    X = np.asarray(X)

    col = X[:, idx]
    unique_vals = np.unique(col)
    if values is None:
        if unique_vals.size <= 5:
            values = unique_vals.tolist()
        else:
            qs = np.linspace(0, 1, 5)
            values = np.quantile(col, qs).tolist()


    # Build a “mean” template for X
    mean_vec = X.mean(axis=0, keepdims=True)  # shape (1, p)
    n = X.shape[0]

    if ax is None:
        fig, ax = plt.subplots()
    for v in values:
        Xv = np.tile(mean_vec, (n, 1))
        Xv[:, idx] = v

        # compute survival curves: S0(t)**exp(pred_i)
        lp = predict_fn(Xv)
        surv = baseline_survival[:, None] ** np.exp(lp)
        curve = surv.mean(axis=1)

        ax.plot(durations, curve, label=f"{round(v,3)}", **plot_kwargs)

    ax.set_title(f"Survival Curves by {feature_name}")
    ax.set_xlabel("Time")
    ax.set_ylabel("Survival Probability")
    ax.legend(title=feature_name)
    return ax

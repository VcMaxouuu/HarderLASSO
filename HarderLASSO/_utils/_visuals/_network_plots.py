"""Basic neural network visualization utilities."""

import numpy as np
import matplotlib.pyplot as plt

def plot_lollipop(
        coefficients,
        feature_names,
        sorted_by_magnitude=False,
        figsize=(8, 6),
        save_path=None
    ):
    """
    Plot a lollipop chart of feature coefficients.

    Parameters
    ----------
    coefficients : np.ndarray
        Coefficients of the features.
    feature_names : list
        Names of the features corresponding to the coefficients.
    sorted_by_magnitude : bool, default=False
        If True, sort the coefficients by their absolute values.
    figsize : tuple, default=(8, 6)
        Size of the figure.
    save_path : str, optional
        Path to save the plot. If None, the plot is not saved.

    Returns
    -------
    matplotlib.axes.Axes
        The axes object containing the plot.

    Raises
    ------
    ValueError
        If the number of features in weight does not match feature names.
    """
    if len(coefficients) != len(feature_names):
        raise ValueError("The number of features in weight does not match the number of feature names.")

    if sorted_by_magnitude:
        order = np.argsort(np.abs(coefficients))
        coefficients, feature_names = coefficients[order], [feature_names[i] for i in order]

    y = np.arange(len(coefficients))
    fig, ax = plt.subplots(figsize=figsize)
    ax.hlines(y, 0, coefficients, alpha=0.3)
    ax.plot(coefficients, y, "o", markersize=7)
    ax.set_yticks(y)
    ax.set_yticklabels(feature_names)
    ax.set_ylabel("Feature")
    ax.set_xlabel("Coefficient weight")

    title = "Feature importance" if sorted else "Non-zero Coefficients Visualization"
    ax.set_title(title)
    if save_path:
        plt.savefig(save_path)
    return ax

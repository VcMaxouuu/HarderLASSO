import numpy as np
import matplotlib.pyplot as plt

def plot_lollipop(coefs, features_names, sorted=False, save_path=None,  figsize=(8,6)):
    if len(coefs) != len(features_names):
        raise ValueError("The number of features in weight does not match the number of feature names.")

    if sorted:
        order = np.argsort(np.abs(coefs))
        coefs, features_names = np.abs(coefs[order]), [features_names[i] for i in order]

    y = np.arange(len(coefs))
    fig, ax = plt.subplots(figsize=figsize)
    ax.hlines(y, 0, coefs, alpha=0.3)
    ax.plot(coefs, y, "o", markersize=7)
    ax.set_yticks(y)
    ax.set_yticklabels(features_names)
    ax.set_ylabel("Feature")
    ax.set_xlabel("Coefficient weight")

    title = "Feature importance" if sorted else "Non-zero Coefficients Visualization"
    ax.set_title(title)
    if save_path:
        plt.savefig(save_path)
    return ax

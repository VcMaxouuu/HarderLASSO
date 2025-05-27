import matplotlib.pyplot as plt

def plot_actual_vs_predicted(y_pred, y_true, ax=None,  **plot_kwargs):
    if ax is None:
        fig, ax = plt.subplots()

    ax.scatter(y_true, y_pred, **plot_kwargs)
    ax.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], linestyle='--', color='red')
    ax.set_title("Actual vs Predicted")
    ax.set_xlabel("Actual")
    ax.set_ylabel("Predicted")
    return ax

def plot_residuals_vs_predicted(y_pred, y_true, ax=None,  **plot_kwargs):
    residuals = y_true - y_pred
    if ax is None:
        fig, ax = plt.subplots()
    ax.scatter(y_pred, residuals, **plot_kwargs)
    ax.axhline(0, ls='--', color='red')
    ax.set_title("Residuals vs Predicted")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Residuals")
    return ax

def plot_residual_distribution(y_pred, y_true, ax=None,  **plot_kwargs):
    residuals = y_true - y_pred
    if ax is None:
        fig, ax = plt.subplots()
    ax.hist(residuals, density=True, **plot_kwargs)
    ax.set_title("Residuals Distribution")
    ax.set_xlabel("Residuals")
    return ax

def plot_qq(y_pred, y_true, ax=None,  **plot_kwargs):
    import scipy.stats as stats

    residuals = y_true - y_pred
    if ax is None:
        fig, ax = plt.subplots()

    # Generate QQ plot manually so we can pass kwargs
    (osm, osr), (slope, intercept, r) = stats.probplot(residuals, dist="norm")
    ax.scatter(osm, osr, **plot_kwargs)
    ax.plot(osm, slope * osm + intercept, color='red', linestyle='--')
    ax.set_title("QQ-Plot of Residuals")
    ax.set_xlabel("Theoretical Quantiles")
    ax.set_ylabel("Sample Quantiles")
    return ax

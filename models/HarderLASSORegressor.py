from .base import HarderLASSOModel
from tasks import RegressorTaskMixin
from QUT import RegressionQUT

class HarderLASSORegressor(
    HarderLASSOModel,
    RegressorTaskMixin
):
    def __init__(
        self,
        hidden_dims: tuple[int, ...] = (20,),
        lambda_qut: float | None = None,
        nu: float = 0.1
    ):
        HarderLASSOModel.__init__(
            self,
            hidden_dims = hidden_dims,
            output_dim = 1,
            bias = True,
            lambda_qut = lambda_qut,
            nu = nu
        )

        RegressorTaskMixin.__init__(self)

        self.QUT = RegressionQUT(
            lambda_qut = lambda_qut,
        )

    def plot_diagnostics(self, X, y):
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))

        self.plot_actual_vs_predicted(X, y, ax=axes[0, 0])
        self.plot_residuals_vs_predicted(X, y, ax=axes[0, 1])
        self.plot_residual_distribution(X, y, ax=axes[1, 0])
        self.plot_qq(X, y, ax=axes[1, 1])

        plt.tight_layout()
        plt.show()

    def plot_actual_vs_predicted(self, X, y, ax=None, **plot_kwargs):
        from utils.visuals.regression_plots import plot_actual_vs_predicted

        y_pred = self.predict(X)
        plot_actual_vs_predicted(y_pred, y, ax=ax, **plot_kwargs)

    def plot_residuals_vs_predicted(self, X, y, ax=None, **plot_kwargs):
        from utils.visuals.regression_plots import plot_residuals_vs_predicted

        y_pred = self.predict(X)
        plot_residuals_vs_predicted(y_pred, y, ax=ax, **plot_kwargs)


    def plot_residual_distribution(self, X, y, ax=None, **plot_kwargs):
        from utils.visuals.regression_plots import plot_residual_distribution

        y_pred = self.predict(X)
        plot_residual_distribution(y_pred, y, ax=ax, **plot_kwargs)

    def plot_qq(self, X, y, ax=None, **plot_kwargs):
        from utils.visuals.regression_plots import plot_qq

        y_pred = self.predict(X)
        plot_qq(y_pred, y, ax=ax, **plot_kwargs)

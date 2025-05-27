from .base import *
from tasks import CoxTaskMixin
from QUT import CoxQUT
from utils.cox_utils import *

class HarderLASSOCox(
    HarderLASSOModel,
    CoxTaskMixin,
):
    def __init__(
        self,
        hidden_dims: tuple[int, ...] = (20,),
        lambda_qut: float | None = None,
        nu: float = 0.1,
        approximation: str = 'Breslow'
    ):
        HarderLASSOModel.__init__(
            self,
            hidden_dims = hidden_dims,
            output_dim = 1,
            bias = False,
            lambda_qut = lambda_qut,
            nu = nu
        )

        CoxTaskMixin.__init__(self, approximation=approximation)

        self.QUT = CoxQUT(
            lambda_qut = lambda_qut,
        )

    @property
    def baseline_hazard_(self) -> pd.DataFrame:
        """
        Returns a DataFrame where:
          - index = durations
          - baseline_cumulative_hazard
        """
        if self.selected_features_idx_ is None:
            raise ValueError("Model not trained yet.")
        df = self.training_survival_df_[["durations", "baseline_cumulative_hazard"]]
        df = df.set_index("durations", drop=True)
        return df

    @property
    def baseline_survival_(self) -> pd.DataFrame:
        """
        Returns a DataFrame where:
          - index = durations
          - baseline_survival
        """
        if self.selected_features_idx_ is None:
            raise ValueError("Model not trained yet.")
        df = self.training_survival_df_[["durations", "baseline_survival"]]
        df = df.set_index("durations", drop=True)
        return df

    @property
    def survival_functions_(self) -> pd.DataFrame:
        """
        Returns a DataFrame where:
          - index = durations
          - each column is 'individual_i' giving S_i(t)
        """
        if self.selected_features_idx_ is None:
            raise ValueError("Model not trained yet.")

        df = self.training_survival_df_
        individual_cols = [c for c in df.columns
                           if c.startswith("individual_")]
        return df.loc[:, ["durations"] + individual_cols].set_index("durations")


    def fit(self, *args, **kwargs):
        super().fit(*args, **kwargs)

        X, (times, events), *rest = args
        preds = self.predict(X)

        self.training_survival_df_ = survival_analysis_training_info(
            times, events, preds
        )


    def plot_baseline_hazard(self, ax=None, **plot_kwargs):
        """
        Convenience method: plots the estimated baseline cumulative hazard function.

        Args:
            ax : matplotlib.axes.Axes, optional
                If provided, the plot will be drawn on this axes.
            **plot_kwargs
                Additional keyword arguments passed to `matplotlib.axes.Axes.plot`
                (e.g., `linestyle`, `color`).
        """
        from utils.visuals.cox_plots import plot_baseline_cumulative_hazard
        plot_baseline_cumulative_hazard(self.baseline_hazard_, ax=ax, **plot_kwargs)

    def plot_baseline_survival(self, ax=None, **plot_kwargs):
        """
        Convenience method: plots the estimated baseline survival function.

        Args:
            ax : matplotlib.axes.Axes, optional
                If provided, the plot will be drawn on this axes.
            **plot_kwargs
                Additional keyword arguments passed to `matplotlib.axes.Axes.plot`
                (e.g., `linestyle`, `color`).
        """
        from utils.visuals.cox_plots import plot_baseline_survival_function
        plot_baseline_survival_function(self.baseline_survival_, ax=ax, **plot_kwargs)

    def plot_survival(
        self,
        predictions: np.ndarray | None = None,
        individuals_idx: int | list[int] | None = None,
        ax=None,
        **plot_kwargs
    ):
        """
        Plot individual survival curves.

        If `predictions` is provided, computes new curves via
        S_i(t) = S₀(t) ** exp(predictions[i]); otherwise it uses
        the precomputed training curves in `self.survival_functions_`.

        Parameters
        ----------
        predictions : array-like of shape (n_individuals,), optional
            Log-hazard predictions for each individual. If None,
            uses the stored survival functions.
        individuals_idx : int or list of int, optional
            Index (0-based) or list of indices of individuals to plot.
            If None, plots all individuals.
        ax : matplotlib.axes.Axes, optional
            If provided, the plot will be drawn on this axes.
        **plot_kwargs
            Additional keyword arguments passed to `matplotlib.axes.Axes.plot`
            (e.g., `linestyle`, `color`).
        """
        from utils.visuals.cox_plots import plot_survival_curves
        durations = self.baseline_survival_.index.values

        if predictions is not None:
            baseline = self.baseline_survival_["baseline_survival"].values
            preds = np.asarray(predictions)
            survival_matrix = baseline[:, None] ** np.exp(preds)
        else:
            sf_df = self.survival_functions_
            survival_matrix = sf_df.values

        labels = list(self.survival_functions_.columns)

        # delegate
        plot_survival_curves(
            durations=durations,
            survival_matrix=survival_matrix,
            labels=labels,
            individuals_idx=individuals_idx,
            ax=ax,
            **plot_kwargs
        )


    def plot_feature_effects(
        self,
        feature_name: str,
        X: np.ndarray,
        values: list | None = None,
        ax=None,
        **plot_kwargs
    ):
        """
        Plot how varying `feature_name` affects survival, holding all other
        features at their mean.

        Parameters
        ----------
        feature_name : str
            The feature to vary.
        X : array-like, shape (n_samples, n_features)
            Covariate matrix to use.
        values : list of scalars, optional
            Which values of `feature_name` to plot.
            If None, uses all unique values when there are ≤5,
            otherwise uses the [0%,25%,50%,75%,100%] quantiles.
        ax : matplotlib.axes.Axes, optional
            If provided, the plot will be drawn on this axes.
        **plot_kwargs
            Passed to the underlying plot (e.g. color, linestyle).
        """
        df = self.training_survival_df_
        durations = df["durations"].values
        baseline_surv = df["baseline_survival"].values

        if feature_name not in self.input_features_:
            raise ValueError(f"{feature_name!r} not in feature_names")

        idx = self.input_features_.index(feature_name)

        from utils.visuals.cox_plots import plot_feature_effects

        plot_feature_effects(
            durations=durations,
            baseline_survival=baseline_surv,
            X=X,
            predict_fn=self.predict,
            feature_name=feature_name,
            idx=idx,
            values=values,
            ax=ax,
            **plot_kwargs
        )

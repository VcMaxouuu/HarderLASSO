import numpy as np
import os
import sys
import pandas as pd
from lifelines import CoxPHFitter
from sksurv.linear_model import CoxnetSurvivalAnalysis
from sklearn.metrics import make_scorer
from sksurv.util import Surv
from sklearn.model_selection import GridSearchCV, KFold
from sksurv.metrics import concordance_index_censored

sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
from HarderLASSO import HarderLASSOCox
import warnings
from multiprocessing import Pool

warnings.filterwarnings('ignore')

def cindex_scorer(y_true, y_pred):
    event = y_true["event"].astype(bool)
    time = y_true["time"].astype(float)
    cindex = concordance_index_censored(event, time, y_pred)[0]
    return cindex

cindex_scorer = make_scorer(cindex_scorer, greater_is_better=True)

class LassoCV:
    def __init__(self,  cv=5):
        self.selected_features_indices_ = []
        self.cv = cv
        self.alphas = np.logspace(-3, 1, 20)
        self.best_model = None

    def fit(self, X, target):
        X = np.asarray(X)
        target = np.asarray(target)
        time = target[0]
        event = target[1]

        y = Surv.from_arrays(event=event.astype(bool), time=time)

        estimator = CoxnetSurvivalAnalysis(l1_ratio=1.0)

        # Cross-validation
        cv = KFold(n_splits=self.cv, shuffle=True)
        gcv = GridSearchCV(
            estimator,
            param_grid={"alphas": [[a] for a in self.alphas]},
            cv=cv,
            scoring=cindex_scorer,
            refit=True,
        )
        gcv.fit(X, y)

        # Store results
        self.best_model = gcv.best_estimator_
        self.selected_features_indices_ = np.nonzero(self.best_model.coef_)[0].tolist()

        return self

    def predict(self, X):
        if self.best_model is None:
            raise RuntimeError("Model not fitted yet.")
        return self.best_model.predict(X)


class ICCox:
    """Greedy forward stepwise subset selection for Cox PH by AIC/BIC."""
    def __init__(self, criterion='bic'):
        self.criterion = criterion
        self.best_model = None
        self.best_ic = None
        self.selected_features = []

    def _ic_from_loglik(self, loglik, k, n_uncensored):
        if self.criterion.lower() == "aic":
            return -2.0 * loglik + 2.0 * k
        elif self.criterion.lower() == "bic":
            return -2.0 * loglik + k * np.log(n_uncensored)
        else:
            raise ValueError("criterion must be 'aic' or 'bic'")

    def _fit_cox_ic(self, df, duration_col, event_col, features):
        try:
            cph = CoxPHFitter()
            cph.fit(df[[duration_col, event_col] + features], duration_col=duration_col, event_col=event_col)
            loglik = cph.log_likelihood_
            k = len(cph.params_)
            n_uncensored = df[event_col].sum()
            return cph, loglik, k, n_uncensored
        except Exception:
            return None, -np.inf, 0, df[event_col].sum()

    def fit(self,  X, target):
        df = pd.DataFrame(X)
        df['duration'] = target[0]
        df['event'] = target[1]

        candidate_features = df.drop(columns=['duration', 'event']).columns.tolist()
        features = []

        def current_ic(feats):
            model, ll, k, n = self._fit_cox_ic(df, duration_col='duration', event_col='event', features=feats)
            return self._ic_from_loglik(ll, k, n), model

        improved = True
        best_ic, best_model = current_ic(features)
        while improved:
            improved = False
            trial_results = []

            remaining = [x for x in candidate_features if x not in features]
            if len(remaining) == 0:
                break
            for f in remaining:
                trial = features + [f]
                ic_val, model = current_ic(trial)
                trial_results.append((ic_val, model, trial))

            if not trial_results:
                break

            ic_min, model_min, feats_min = min(trial_results, key=lambda z: z[0])

            if ic_min < best_ic:
                best_ic, best_model, features = ic_min, model_min, feats_min
                improved = True

        self.selected_features = features
        self.best_ic = best_ic
        self.best_model = best_model
        self.selected_features_indices_ = [df.drop(columns=['duration', 'event']).columns.get_loc(f) for f in self.selected_features]

    def predict(self, X):
        if self.best_model is None:
            raise RuntimeError("Model not fitted yet.")

        if len(self.selected_features_indices_) == 0:
            return np.zeros(X.shape[0])

        X = X[:, self.selected_features_indices_]
        return self.best_model.predict_partial_hazard(X)


def run_single_simulation(args):
    """Run a single simulation for given parameters"""
    X_train, X_test, selected_indices, s_value, sim_idx = args
    np.random.seed(42 + sim_idx)

    if s_value == 0:
        t_train = np.exp(np.zeros(X_train.shape[0]))
        t_test = np.exp(np.zeros(X_test.shape[0]))
    else:
        X_train_s = X_train[:, selected_indices]
        X_test_s = X_test[:, selected_indices]
        beta = np.random.choice(np.array([i for i in range(-3, 3) if i != 0]), size=s_value, replace=True)
        t_train = np.exp(X_train_s@beta)
        t_test = np.exp(X_test_s@beta)

    T_train = np.random.exponential(scale=1/t_train)
    T_test = np.random.exponential(scale=1/t_test)

    C = np.quantile(T_train, 0.5)

    y_train = np.minimum(T_train, C)
    y_test = np.minimum(T_test, C)
    c_train = T_train <= C
    c_test = T_test <= C

    # Fit models
    methods = {
        'LASSO_CV': LassoCV(cv=5),
        'AIC_Cox': ICCox(criterion='aic'),
        'BIC_Cox': ICCox(criterion='bic'),
        'LASSO_QUT': HarderLASSOCox(penalty='l1', hidden_dims=None),
        'HarderLASSO_QUT': HarderLASSOCox(penalty='harder', hidden_dims=None),
    }

    results = {}

    for name, method in methods.items():
        try:
            method.fit(X_train, (y_train, c_train))

            if hasattr(method, 'selected_features_indices_'):
                selected_features = method.selected_features_indices_
            else:
                selected_features = np.nonzero(method.coef_)[0].tolist()

            c_index = concordance_index_censored(c_test, y_test, method.predict(X_test))[0]

            results[name] = {
                'c_index': c_index,
                'selected_features': selected_features
            }
        except Exception as e:
            print(f"Error with {name} for s={s_value}, sim={sim_idx}: {e}")
            results[name] = {
                'c_index': np.inf,
                'selected_features': []
            }

    return results


def main():
    np.random.seed(42)

    print("Starting linear simulation...")

    n_train = 150
    n_test = 1000
    n_features = 100
    n_simulations = 200

    print("Generating training and testing matrices...")
    X_train = np.random.randn(n_train, n_features)
    X_test = np.random.randn(n_test, n_features)

    X_train = (X_train - np.mean(X_train, axis=0)) / np.std(X_train, axis=0)
    X_test = (X_test - np.mean(X_test, axis=0)) / np.std(X_test, axis=0)

    os.makedirs('simulations/survival/linear/column_selections', exist_ok=True)
    os.makedirs('simulations/survival/linear/results', exist_ok=True)

    print("Generating column selections...")
    s_range = range(21)
    for s in s_range:
        if s == 0:
            selections_df = pd.DataFrame({'simulation': range(n_simulations), 'selected_columns': ['' for _ in range(n_simulations)]})
        else:
            selections = []
            for sim_idx in range(n_simulations):
                selected_cols = np.random.choice(n_features, size=s, replace=False)
                selections.append(','.join(map(str, selected_cols)))
            selections_df = pd.DataFrame({'simulation': range(n_simulations), 'selected_columns': selections})

        selections_df.to_csv(f'simulations/survival/linear/column_selections/s_{s}.csv', index=False)

    print("Running simulations...")

    for s in s_range:
        print(f"Processing s = {s}")

        selections_df = pd.read_csv(f'simulations/survival/linear/column_selections/s_{s}.csv')

        args_list = []
        for sim_idx in range(n_simulations):
            if s == 0:
                selected_cols = []
            else:
                col_data = selections_df.loc[sim_idx, 'selected_columns']
                if pd.isna(col_data) or col_data == '':
                    selected_cols = []
                elif isinstance(col_data, (int, np.integer)):
                    selected_cols = [int(col_data)]
                else:
                    selected_cols = list(map(int, str(col_data).split(',')))

            args_list.append((
                X_train, X_test,
                selected_cols, s, sim_idx
            ))

        with Pool() as pool:
            simulation_results = pool.map(run_single_simulation, args_list)

        # Create results DataFrame with 100 rows and method columns
        results_data = []
        for sim_idx, result in enumerate(simulation_results):
            row = {'simulation': sim_idx}
            for method_name, method_result in result.items():
                features_str = ','.join(map(str, method_result['selected_features']))
                row[f'{method_name}_c_index'] = method_result['c_index']
                row[f'{method_name}_features'] = features_str
            results_data.append(row)

        results_df = pd.DataFrame(results_data)
        results_df.to_csv(f'simulations/survival/linear/results/s_{s}_results.csv', index=False)

    print("Simulation completed!")


if __name__ == "__main__":
    main()

import numpy as np
import pandas as pd

from lifelines import CoxPHFitter
from sksurv.linear_model import CoxnetSurvivalAnalysis
from sklearn.metrics import make_scorer
from sksurv.util import Surv
from sklearn.model_selection import GridSearchCV, KFold, train_test_split
from sksurv.metrics import concordance_index_censored
from sklearn.preprocessing import StandardScaler

from multiprocessing import Pool
from pathlib import Path
import os
import sys

import warnings
warnings.filterwarnings('ignore')

sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
from HarderLASSO import HarderLASSOCox


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
        cv = KFold(n_splits=self.cv)
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
        df['duration'] = target[0].values
        df['event'] = target[1].values

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
        X = np.asarray(X)
        if self.best_model is None:
            raise RuntimeError("Model not fitted yet.")

        if len(self.selected_features_indices_) == 0:
            return np.zeros(X.shape[0])

        X = X[:, self.selected_features_indices_]
        return self.best_model.predict_partial_hazard(X)


_X = None
_y = None
_c = None

def _init_worker(X, y, c):
    global _X, _y, _c
    _X, _y, _c = X, y, c

def run_single_simulation(seed):
    np.random.seed(seed)

    X_train, X_test, y_train, y_test, c_train, c_test = train_test_split(
        _X, _y, _c, train_size=2/3, random_state=seed, shuffle=True, stratify=_c
    )

    methods = {
        'HarderLASSO_QUT': HarderLASSOCox(hidden_dims=None, penalty='harder'),
        'LASSO_QUT': HarderLASSOCox(hidden_dims=None, penalty='l1'),
        'AIC': ICCox(criterion='aic'),
        'BIC': ICCox(criterion='bic'),
        'LASSO_CV': LassoCV(cv=5),
    }

    results = {}

    for name, method in methods.items():
        try:
            method.fit(X_train, (y_train, c_train))

            if hasattr(method, 'selected_features_indices_'):
                selected_features = method.selected_features_indices_
            else:
                selected_features = np.nonzero(method.coef_)[0].tolist()

            c_index = concordance_index_censored(c_test.astype(bool), y_test, method.predict(X_test))[0]

            results[name] = {
                'c_index': c_index,
                'selected_features': selected_features
            }
        except Exception as e:
            print(f"Error with {name}, sim={seed}: {e}")
            results[name] = {
                'c_index': np.inf,
                'selected_features': []
            }

    return results


def main():
    np.random.seed(42)
    n_simulations = 100

    print("Loading PBC dataset...")
    df = pd.read_csv('simulations/survival/PBC/data.csv').drop(columns=['id'])
    df['status'] = df['status'].replace({0: 0, 1: 0, 2: 1})
    df = df.dropna()
    df['sex'] = df['sex'].replace({'m': 0, 'f': 1})

    X = df.drop(columns=['time', 'status'])
    c = df['status'].astype(bool)
    y = df['time']

    X = StandardScaler().fit_transform(X)

    seeds = list(range(n_simulations))
    with Pool(initializer=_init_worker, initargs=(X, y, c)) as pool:
        simulation_results = pool.map(run_single_simulation, seeds)

    results_data = []
    for sim_idx, result in enumerate(simulation_results):
        row = {'simulation': sim_idx}
        for method_name, method_result in result.items():
            row[f'{method_name}_c_index'] = method_result['c_index']
            row[f'{method_name}_features'] = ','.join(map(str, method_result['selected_features']))
        results_data.append(row)

    results_df = pd.DataFrame(results_data)
    results_df.to_csv("simulations/survival/PBC/results.csv", index=False)

    print("Simulation completed!")


if __name__ == "__main__":
    main()

import numpy as np
import pandas as pd
from lifelines import CoxPHFitter
from sksurv.linear_model import CoxnetSurvivalAnalysis
from sklearn.metrics import make_scorer
from SurvSet.data import SurvLoader
from sksurv.util import Surv
from sksurv.datasets import load_gbsg2, load_whas500, load_flchain
from sklearn.model_selection import GridSearchCV, KFold, train_test_split
from sksurv.metrics import concordance_index_censored
from sklearn.preprocessing import StandardScaler
from ucimlrepo import fetch_ucirepo
from lassonet import LassoNetCoxRegressorCV
from sksurv.ensemble import GradientBoostingSurvivalAnalysis
from boruta import BorutaPy
from multiprocessing import Pool
from pathlib import Path
import os
import sys
import warnings
import statsmodels.api as sm
warnings.filterwarnings('ignore')

sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
from HarderLASSO import HarderLASSOCox


# ---------------------------------------------------------------------
# Dataset loading functions
# ---------------------------------------------------------------------
def load_ovarian_data():
    df, ref = SurvLoader().load_dataset(ds_name='dataOvarian1').values()
    X = df.drop(columns=['pid', 'event', 'time'])
    c = df['event']
    t = df['time']
    return X, t, c

def load_nsbcd_data():
    df, ref = SurvLoader().load_dataset(ds_name='NSBCD').values()
    X = df.drop(columns=['pid', 'event', 'time'])
    c = df['event']
    t = df['time']
    return X, t, c

def load_mcl_data():
    df, ref = SurvLoader().load_dataset(ds_name='MCLcleaned').values()
    X = df.drop(columns=['pid', 'event', 'time'])
    c = df['event']
    t = df['time']
    return X, t, c

def load_nki70_data():
    df, ref = SurvLoader().load_dataset(ds_name='nki70').values()
    X = df.drop(columns=['pid', 'event', 'time'])
    c = df['event']
    t = df['time']
    X['fac_Diam'] = df['fac_Diam'].map({'<=2cm': 0, '>2cm': 1})
    X['fac_N'] = df['fac_N'].map({'1-3': 0, '>=4': 1})
    X['fac_ER'] = df['fac_ER'].map({'Negative': 0, 'Positive': 1})
    X['fac_Grade'] = df['fac_Grade'].map({'Poorly diff': 0, 'Intermediate': 1, 'Well diff': 2})

    return X, t, c

def load_rhc_data():
    df, ref = SurvLoader().load_dataset(ds_name='rhc').values()
    X = df.drop(columns=['pid', 'event', 'time', 'num_adld3p', 'num_urin1'])
    c = df['event']
    t = df['time']
    X = pd.get_dummies(X, drop_first=True)
    return X, t, c

def load_gbsg2_data():
    X, y = load_gbsg2()
    X['horTh'] = X['horTh'].map({'no': 0, 'yes': 1})
    X['menostat'] = X["menostat"].map({'Pre': 0, 'Post': 1})
    X = pd.get_dummies(X, columns=["tgrade"], drop_first=True)
    c = y['cens']
    t = y['time']
    return X, t, c

def load_pbc_data():
    data = sm.datasets.get_rdataset("pbc", package="survival")
    df = data.data
    df = df.drop(columns=['id'])
    df['status'] = df['status'].replace({0: 0, 1: 0, 2: 1})
    df = df.dropna()
    df['sex'] = df['sex'].replace({'m': 0, 'f': 1})
    X = df.drop(columns=['time', 'status'])
    t = df['time']
    c = df['status']
    return X, t, c

def load_whas_data():
    X, y = load_whas500()
    c = y['fstat']
    t = y['lenfol']
    return X, t, c


def load_flchain_data():
    X, y = load_flchain()
    y = pd.DataFrame(y, columns=['futime', 'death'], index=X.index)
    df = pd.concat([X, y], axis=1)
    df = df.drop(columns=['chapter'], errors='ignore')
    df['mgus'] = df['mgus'].map({'no': 0, 'yes': 1})
    df['sex'] = df['sex'].map({'F': 0, 'M': 1})
    s = df.select_dtypes(include='category').columns
    df[s] = df[s].astype("float")
    df = df.fillna(df.mean())
    t = df['futime']
    c = df['death']
    X = df.drop(columns=['futime', 'death'])
    return X, t, c


def load_support_data():
    support2 = fetch_ucirepo(id=880)
    X = support2.data.features
    c = support2.data.targets['death']
    t = support2.data.original['d.time']
    to_keep_columns = [
        "age", "sex", "race", "num.co", "diabetes", "dementia", "ca",
        "meanbp", "hrt", "resp", "temp", "wblc", "sod", "crea"
    ]

    X = X[to_keep_columns].copy()
    data = pd.concat([X, t, c], axis=1)
    data = data.dropna()
    X = data[to_keep_columns].copy()
    c = data[c.name]
    t = data[t.name]
    X["sex"] = X["sex"].map({"male": 0, "female": 1})
    X = pd.get_dummies(X, columns=["race", "ca", "num.co"], drop_first=True)
    return X, t, c


# ---------------------------------------------------------------------
# Scoring and model wrappers
# ---------------------------------------------------------------------
def cindex_scorer(y_true, y_pred):
    event = y_true["event"].astype(bool)
    time = y_true["time"].astype(float)
    cindex = concordance_index_censored(event, time, y_pred)[0]
    return cindex

cindex_scorer = make_scorer(cindex_scorer, greater_is_better=True)


class LassoCV:
    def __init__(self, cv=5):
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

        cv = KFold(n_splits=self.cv)
        gcv = GridSearchCV(
            estimator,
            param_grid={"alphas": [[a] for a in self.alphas]},
            cv=cv,
            scoring=cindex_scorer,
            refit=True,
        )
        gcv.fit(X, y)
        self.best_model = gcv.best_estimator_
        self.selected_features_indices_ = np.nonzero(self.best_model.coef_)[0].tolist()
        return self

    def predict(self, X):
        if self.best_model is None:
            raise RuntimeError("Model not fitted yet.")
        return self.best_model.predict(X)


class ICCox:
    def __init__(self, criterion='bic', n_jobs=None):
        self.criterion = criterion
        self.n_jobs = n_jobs
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
            cph.fit(df[[duration_col, event_col] + features],
                    duration_col=duration_col, event_col=event_col)
            loglik = cph.log_likelihood_
            k = len(cph.params_)
            n_uncensored = df[event_col].sum()
            return cph, loglik, k, n_uncensored
        except Exception:
            return None, -np.inf, 0, df[event_col].sum()

    def fit(self, X, target):
        df = pd.DataFrame(X)
        df['duration'] = np.asarray(target[0])
        df['event'] = np.asarray(target[1])
        candidate_features = df.drop(columns=['duration', 'event']).columns.tolist()
        features = []

        def current_ic(feats):
            model, ll, k, n = self._fit_cox_ic(df, 'duration', 'event', feats)
            return self._ic_from_loglik(ll, k, n), model

        improved = True
        best_ic, best_model = current_ic(features)
        while improved:
            improved = False
            trial_results = []
            remaining = [x for x in candidate_features if x not in features]
            if len(remaining) == 0:
                break

            with Pool(processes=self.n_jobs) as pool:
                trial_results = pool.map(
                    lambda f: current_ic(features + [f]),
                    remaining
                )

            if not trial_results:
                break
            ic_min, model_min, feats_min = min(trial_results, key=lambda z: z[0])
            if ic_min < best_ic:
                best_ic, best_model, features = ic_min, model_min, feats_min
                improved = True

        self.selected_features = features
        self.best_ic = best_ic
        self.best_model = best_model
        self.selected_features_indices_ = [
            df.drop(columns=['duration', 'event']).columns.get_loc(f) for f in self.selected_features
        ]

    def predict(self, X):
        if self.best_model is None:
            raise RuntimeError("Model not fitted yet.")
        if len(self.selected_features_indices_) == 0:
            return np.zeros(X.shape[0])
        df = pd.DataFrame(X[:, self.selected_features_indices_],
                          columns=self.selected_features)
        return self.best_model.predict_partial_hazard(df)


class LassoNetFeatureSelector:
    """LASSO-Net with automatic feature selection through regularization path"""
    def __init__(self, hidden_dims=(20,), cv=5):
        self.hidden_dims = hidden_dims
        self.cv = cv
        self.selected_features_indices_ = []
        self.model = None

    def fit(self, X, target):
        target = np.asarray(target, dtype=np.float32).T
        self.model = LassoNetCoxRegressorCV(
            tie_approximation='breslow',
            hidden_dims=self.hidden_dims,
            cv=self.cv,
            verbose=0
        )
        self.model.fit(X, target)

        self.selected_features_indices_ = np.nonzero(self.model.best_selected_.numpy())[0].tolist()

    def predict(self, X):
        return self.model.predict(X).squeeze()

class GBoostFeatureSelector:
    """Gradient Boosting with Boruta feature selection"""
    def __init__(self):
        self.model = GradientBoostingSurvivalAnalysis()
        self.selected_features_indices_ = []
        self.boruta_selector = None

    def fit(self, X, target):
        X = np.array(X, dtype=float)
        target = np.array([(e, t) for t, e in zip(target[0], target[1])],
                          dtype=[('event', bool), ('time', float)])

        self.boruta_selector = BorutaPy(
            estimator=self.model,
            n_estimators='auto',
            verbose=0
        )
        self.boruta_selector.fit(X, target)
        self.selected_features_indices_ = np.where(self.boruta_selector.support_)[0].tolist()

        if self.selected_features_indices_:
            X_sel = X[:, self.selected_features_indices_]
            self.model.fit(X_sel, target)

    def predict(self, X):
        if len(self.selected_features_indices_) > 0:
            X_selected = X[:, self.selected_features_indices_]
            return self.model.predict(X_selected)
        else:
            return np.zeros(X.shape[0])


class LinearCoxModel:
    def __init__(self):
        self.model = CoxPHFitter()
        self.selected_features_indices_ = []

    def fit(self, X, target):
        df = pd.DataFrame(X)
        df['duration'] = np.asarray(target[0])
        df['event'] = np.asarray(target[1])
        self.model.fit(df, duration_col='duration', event_col='event')

    def predict(self, X):
        df = pd.DataFrame(X)
        return self.model.predict_partial_hazard(df).values



# ---------------------------------------------------------------------
# Parallel simulation engine
# ---------------------------------------------------------------------
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
        #'HarderLASSO_QUT': HarderLASSOCox(hidden_dims=None, penalty='harder'),
        #'LASSO_QUT': HarderLASSOCox(hidden_dims=None, penalty='l1'),
        #'HarderLASSO_QUT_ANN': HarderLASSOCox(hidden_dims=(20, ), penalty='harder'),
        #'LASSO_QUT_ANN': HarderLASSOCox(hidden_dims=(20, ), penalty='l1'),
        #'AIC': ICCox(criterion='aic'),
        #'BIC': ICCox(criterion='bic'),
        #'LASSO_CV': LassoCV(cv=5),
        #'LassoNet': LassoNetFeatureSelector(hidden_dims=(20, ), cv=5),
        #'CoxPH': LinearCoxModel(),
        'GBoost': GBoostFeatureSelector(),
    }

    results = {}
    for name, method in methods.items():
        try:
            method.fit(X_train, (y_train, c_train))
            preds = method.predict(X_test)
            c_index = concordance_index_censored(c_test.astype(bool), y_test, preds)[0]
            selected_features = getattr(method, 'selected_features_indices_', [])
            results[name] = {'c_index': c_index, 'selected_features': selected_features}
        except Exception as e:
            print(f"Error with {name}, sim={seed}: {e}")
            results[name] = {'c_index': np.nan, 'selected_features': []}
    return results


# ---------------------------------------------------------------------
# Main simulation loop over datasets
# ---------------------------------------------------------------------
def main():
    np.random.seed(42)
    n_simulations = 100
    datasets = {
        #"GBSG2": load_gbsg2_data,
        #"WHAS500": load_whas_data,
        #"FLCHAIN": load_flchain_data,
        #"SUPPORT": load_support_data,
        #"PBC": load_pbc_data,
        #"OVARIAN": load_ovarian_data,
        #"NSBCD": load_nsbcd_data,
        "MCL": load_mcl_data,
        #"NKI70": load_nki70_data,
        #"RHC": load_rhc_data
    }

    output_dir = Path("simulations/survival/classical_datasets/results")
    output_dir.mkdir(parents=True, exist_ok=True)

    dataset_summary = []

    for name, loader in datasets.items():
        print(f"\n=== Running simulations for {name} dataset ===")
        X, y, c = loader()
        X = pd.DataFrame(X)
        y = np.asarray(y)
        c = np.asarray(c)
        X = StandardScaler().fit_transform(X)

        n, p = X.shape
        n_uncensored = int(c.sum())  # number of observed events
        dataset_summary.append({
            "dataset": name,
            "n_samples": n,
            "n_features": p,
            "n_uncensored": n_uncensored
        })

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
        results_df.to_csv(output_dir / f"{name}_results.csv", index=False)
        print(f"Saved results to {output_dir / f'{name}_results.csv'}")

    summary_df = pd.DataFrame(dataset_summary)
    summary_path = output_dir / "dataset_summary.csv"
    summary_df.to_csv(summary_path, index=False)
    print(f"\nDataset summary saved to {summary_path}")

    print("\nAll simulations completed!")


if __name__ == "__main__":
    main()

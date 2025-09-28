import numpy as np
import os
import sys
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegressionCV

from boruta import BorutaPy
import xgboost as xgb
from lassonet import LassoNetClassifierCV

from multiprocessing import Pool
from pathlib import Path
import warnings
from collections import Counter

warnings.filterwarnings('ignore')

sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
from HarderLASSO import HarderLASSOClassifier

class LassoClassifierCV:
    def __init__(self, cv=5):
        self.clf = LogisticRegressionCV(
                penalty="l1",
                solver="saga",
                cv=cv,
                scoring='accuracy',
                max_iter=1000,
        )
        self.selected_features_indices_ = []

    def fit(self, X, y):
        self.clf.fit(X, y)
        selected_mask = np.any(self.clf.coef_ != 0, axis=0)
        self.selected_features_indices_ = np.where(selected_mask)[0].tolist()

    def predict(self, X):
        return self.clf.predict(X)


class RandomForestFeatureSelector:
    """Random Forest with Boruta feature selection"""
    def __init__(self):
        self.rf = RandomForestClassifier()
        self.selected_features_indices_ = []
        self.boruta_selector = None
        self.majority_class_ = 0

    def fit(self, X, y):
        self.boruta_selector = BorutaPy(
            estimator=self.rf,
            n_estimators='auto',
            verbose=0
        )
        self.boruta_selector.fit(X, y)
        self.selected_features_indices_ = np.where(self.boruta_selector.support_)[0].tolist()

        if self.selected_features_indices_:
            X_sel = X[:, self.selected_features_indices_]
            self.rf.fit(X_sel, y)
        else:
            counts = Counter(y)
            self.majority_class_ = int(max(counts, key=counts.get))

    def predict(self, X):
        if len(self.selected_features_indices_) > 0:
            X_selected = X[:, self.selected_features_indices_]
            return self.rf.predict(X_selected)
        else:
            return np.full(X.shape[0], self.majority_class_, dtype=int)


class XGBoostFeatureSelector:
    """XGBoost with Boruta feature selection"""

    def __init__(self):
        self.xgb = xgb.XGBClassifier(verbosity=0)
        self.selected_features_indices_ = []
        self.boruta_selector = None
        self.majority_class_ = 0

    def fit(self, X, y):
        self.boruta_selector = BorutaPy(
            estimator=self.xgb,
            n_estimators='auto',
            verbose=0
        )
        self.boruta_selector.fit(X, y)
        self.selected_features_indices_ = np.where(self.boruta_selector.support_)[0].tolist()

        if self.selected_features_indices_:
            X_sel = X[:, self.selected_features_indices_]
            self.xgb.fit(X_sel, y)
        else:
            counts = Counter(y)
            self.majority_class_ = int(max(counts, key=counts.get))

    def predict(self, X):
        if len(self.selected_features_indices_) > 0:
            X_selected = X[:, self.selected_features_indices_]
            return self.xgb.predict(X_selected)
        else:
            return np.full(X.shape[0], self.majority_class_, dtype=int)


class LassoNetFeatureSelector:
    """LASSO-Net with automatic feature selection through regularization path"""
    def __init__(self, hidden_dims=(20,), cv=5):
        self.hidden_dims = hidden_dims
        self.cv = cv
        self.selected_features_indices_ = []
        self.model = None

    def fit(self, X, y):
        self.model = LassoNetClassifierCV(
            hidden_dims=self.hidden_dims,
            cv=self.cv,
            verbose=0
        )
        self.model.fit(X, y)

        self.selected_features_indices_ = np.nonzero(self.model.best_selected_.numpy())[0].tolist()

    def predict(self, X):
        return self.model.predict(X)



_X = None
_y = None

def _init_worker(X, y):
    global _X, _y
    _X, _y = X, y

def run_single_simulation(seed):
    X_train, X_test, y_train, y_test = train_test_split(
        _X, _y, train_size=2/3, random_state=seed, shuffle=True, stratify=_y
    )

    # Fit models
    methods = {
        'LASSO_CV': LassoClassifierCV(cv=5),
        'linear_LASSO_QUT': HarderLASSOClassifier(penalty='l1', hidden_dims=None),
        'linear_HarderLASSO_QUT': HarderLASSOClassifier(penalty='harder', hidden_dims=None),
        'RandomForest': RandomForestFeatureSelector(),
        'XGBoost': XGBoostFeatureSelector(),
        'LassoNet': LassoNetFeatureSelector(hidden_dims=(20,), cv=5),
        'LASSO_QUT': HarderLASSOClassifier(penalty='l1', hidden_dims=(20, )),
        'HarderLASSO_QUT': HarderLASSOClassifier(penalty='harder', hidden_dims=(20, )),
    }

    results = {}

    for name, method in methods.items():
        try:
            method.fit(X_train, y_train)
            y_pred = method.predict(X_test)

            if hasattr(method, 'selected_features_indices_'):
                selected_features = method.selected_features_indices_
            else:
                selected_features = np.nonzero(method.coef_)[0].tolist()

            acc = accuracy_score(y_test, y_pred)

            results[name] = {
                'acc': acc,
                'selected_features': selected_features
            }
        except Exception as e:
            print(f"Error with {name} for seed={seed}: {e}")
            results[name] = {
                'acc': 0,
                'selected_features': []
            }

    return results


def main():
    print("Starting classification simulation...")

    n_simulations = 50
    in_dir = Path("simulations/classification/datasets")
    out_dir = Path("simulations/classification/results")
    out_dir.mkdir(parents=True, exist_ok=True)

    for file in sorted(in_dir.glob("*.csv")):
        print(f"Processing {file}...")
        df = pd.read_csv(file)

        X = df.drop(columns=['label']).to_numpy()
        y = df['label'].to_numpy()

        X = StandardScaler().fit_transform(X)

        seeds = list(range(n_simulations))
        with Pool(initializer=_init_worker, initargs=(X, y)) as pool:
            simulation_results = pool.map(run_single_simulation, seeds, chunksize=1)

        # Create results DataFrame with 100 rows and method columns
        results_data = []
        for sim_idx, result in enumerate(simulation_results):
            row = {'simulation': sim_idx}
            for method_name, method_result in result.items():
                row[f'{method_name}_acc'] = method_result['acc']
                row[f'{method_name}_features'] = ','.join(map(str, method_result['selected_features']))
            results_data.append(row)

        results_df = pd.DataFrame(results_data)
        results_df.to_csv(out_dir / f'{file.stem}_results.csv', index=False)

    print("Simulation completed!")


if __name__ == "__main__":
    main()

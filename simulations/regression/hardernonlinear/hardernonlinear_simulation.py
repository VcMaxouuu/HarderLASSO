import numpy as np
import os
import sys
import pandas as pd
import torch.nn as nn
from sklearn.ensemble import RandomForestRegressor
import xgboost as xgb
from boruta import BorutaPy

sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
from HarderLASSO import HarderLASSORegressor
from sklearn.metrics import mean_squared_error
import warnings
from multiprocessing import Pool

warnings.filterwarnings('ignore')

class RandomForestFeatureSelector:
    """Random Forest with Boruta feature selection"""
    def __init__(self):
        self.rf = RandomForestRegressor()
        self.selected_features_indices_ = []
        self.boruta_selector = None

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
            self.y_mean = np.mean(y)

    def predict(self, X):
        if len(self.selected_features_indices_) > 0:
            X_selected = X[:, self.selected_features_indices_]
            return self.rf.predict(X_selected)
        else:
            return self.y_mean * np.ones(X.shape[0])


class XGBoostFeatureSelector:
    """XGBoost with Boruta feature selection"""

    def __init__(self):
        self.xgb = xgb.XGBRegressor(verbosity=0)
        self.selected_features_indices_ = []
        self.boruta_selector = None

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
            self.y_mean = np.mean(y)

    def predict(self, X):
        if len(self.selected_features_indices_) > 0:
            X_selected = X[:, self.selected_features_indices_]
            return self.xgb.predict(X_selected)
        else:
            return self.y_mean * np.ones(X.shape[0])

def run_single_simulation(args):
    """Run a single simulation for given parameters"""
    X_train, X_test, error_col, selected_inds, n_value, sim_idx = args

    X_train_s = X_train[:, selected_inds]
    X_test_s  = X_test[:, selected_inds]

    mu_train = np.abs(
        10 * np.abs(X_train_s[:, 1] - X_train_s[:, 0])
      - 10 * np.abs(X_train_s[:, 3] - X_train_s[:, 2])
    )

    mu_test = np.abs(
        10 * np.abs(X_test_s[:, 1] - X_test_s[:, 0])
      - 10 * np.abs(X_test_s[:, 3] - X_test_s[:, 2])
    )

    # add noise on training side only
    y_train = mu_train + error_col
    y_test  = mu_test

    # Initialize methods
    methods = {
        'HarderLASSO_(20)': HarderLASSORegressor(hidden_dims=(20, ), activation=nn.ReLU()),
        'HarderLASSO_(20, 10)': HarderLASSORegressor(hidden_dims=(20, 10), activation=nn.ReLU()),
        'HarderLASSO_(20, 10, 5)': HarderLASSORegressor(hidden_dims=(20, 10, 5), activation=nn.ReLU()),

        'RandomForest': RandomForestFeatureSelector(),
        'XGBoost': XGBoostFeatureSelector()
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

            mse = mean_squared_error(y_test, y_pred)

            results[name] = {
                'mse': mse,
                'selected_features': selected_features
            }
        except Exception as e:
            print(f"Error with {name} for n={n_value}, sim={sim_idx}: {e}")
            results[name] = {
                'mse': np.inf,
                'selected_features': []
            }

    return results


def main():
    np.random.seed(42)

    print("Starting nonlinear simulation...")

    s = 4
    n_full = 4000
    n_test = 1000
    n_features = 50
    n_simulations = 500

    print("Generating training and testing matrices...")
    X_full = np.random.randn(n_full, n_features)
    X_full = (X_full - X_full.mean(axis=0)) / X_full.std(axis=0)

    X_test = np.random.randn(n_test, n_features)
    X_test = (X_test - X_test.mean(axis=0)) / X_test.std(axis=0)

    # Create directories for results
    print("Creating directories for results...")
    base_dir       = 'simulations/regression/hardernonlinear'
    selections_dir = os.path.join(base_dir, 'column_selections')
    results_dir    = os.path.join(base_dir, 'results')
    os.makedirs(selections_dir, exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)

    n_values = np.arange(100, n_full + 1, 100)

    for n in n_values:
        print(f"Running simulations for sample size n = {n}")
        X_train = X_full[:n, :]

        selections = [
            np.random.choice(n_features, size=s, replace=False).tolist()
            for _ in range(n_simulations)
        ]
        sel_rows = {
            'simulation': list(range(n_simulations)),
            'selected_columns': [','.join(map(str, sel)) for sel in selections]
        }
        pd.DataFrame(sel_rows).to_csv(
            os.path.join(selections_dir, f'n_{n}.csv'),
            index=False
        )

        error_matrix = np.random.randn(n, n_simulations)

        args_list = [
            (X_train, X_test, error_matrix[:, i], selections[i], n, i)
            for i in range(n_simulations)
        ]

        with Pool() as pool:
            simulation_results = pool.map(run_single_simulation, args_list)

        # Create results DataFrame with 100 rows and method columns
        results_data = []
        for sim_idx, result in enumerate(simulation_results):
            row = {'simulation': sim_idx}
            for method_name, method_result in result.items():
                features_str = ','.join(map(str, method_result['selected_features']))
                row[f'{method_name}_mse'] = method_result['mse']
                row[f'{method_name}_features'] = features_str
            results_data.append(row)

        results_df = pd.DataFrame(results_data)
        results_df.to_csv(os.path.join(results_dir, f'n_{n}_results.csv'), index=False)

    print("Nonlinear simulation completed!")

if __name__ == "__main__":
    main()

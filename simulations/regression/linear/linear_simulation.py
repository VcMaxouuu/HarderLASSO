import numpy as np
import os
import sys
import pandas as pd

import rpy2.robjects as ro
from rpy2.robjects import default_converter
from rpy2.robjects.conversion import localconverter
from rpy2.robjects import numpy2ri
from rpy2.robjects.vectors import FloatVector

sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
from HarderLASSO import HarderLASSORegressor
from sklearn.metrics import mean_squared_error
import warnings
from multiprocessing import Pool

warnings.filterwarnings('ignore')

class CVRegressor:
    def __init__(self, cv=5, penalty='SCAD'):
        numpy2ri.activate()
        ro.r('library(ncvreg)')
        self.family = 'gaussian'
        self.penalty = penalty
        self.cv = cv
        self.cvfit_ = None
        self.coef_ = None
        self.intercept_ = None
        self.selected_features_indices_ = None

    def fit(self, X, y):
        y_r = FloatVector(np.asarray(y, dtype=float).ravel())

        # Safe conversion for X
        with localconverter(default_converter + numpy2ri.converter):
            X_r = ro.conversion.py2rpy(np.asarray(X, dtype=float))

        # Call cv.ncvreg in R
        cvfit = ro.r['cv.ncvreg'](X_r, y_r, family=self.family, penalty=self.penalty, nfolds=self.cv)
        self.cvfit_ = cvfit

        # Extract coefficients at best lambda
        coef_r = ro.r['coef'](cvfit)
        coef_vals = np.array(coef_r)

        self.coef_ = coef_vals[1:]   # exclude intercept
        self.intercept_ = coef_vals[0]

        # Non-zero indices among features
        self.selected_features_indices_ = np.where(self.coef_ != 0)[0]

        return self

    def predict(self, X):
        if self.coef_ is None:
            raise ValueError("Model is not fitted yet. Call fit() first.")
        return self.intercept_ + X.dot(self.coef_)


def run_single_simulation(args):
    """Run a single simulation for given parameters"""
    X_train, X_test, error_col, selected_indices, s_value, sim_idx = args
    np.random.seed(42 + sim_idx)

    if s_value == 0:
        y_train = error_col
        y_test = np.zeros(X_test.shape[0])
    else:
        X_train_s = X_train[:, selected_indices]
        X_test_s = X_test[:, selected_indices]
        beta = np.random.choice(np.array([i for i in range(-3, 3) if i != 0]), size=s_value, replace=True)
        y_train = X_train_s@beta + error_col
        y_test = X_test_s@beta

    # Fit models
    methods = {
        'LASSO_CV': CVRegressor(cv=5, penalty='lasso'),
        'MCP_CV': CVRegressor(cv=5, penalty='MCP'),
        'SCAD_CV': CVRegressor(cv=5, penalty='SCAD'),
        'LASSO_QUT': HarderLASSORegressor(penalty='l1', hidden_dims=None),
        'HarderLASSO_QUT': HarderLASSORegressor(penalty='harder', hidden_dims=None),
        'SCAD_QUT': HarderLASSORegressor(penalty='scad', hidden_dims=None)
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
            print(f"Error with {name} for s={s_value}, sim={sim_idx}: {e}")
            results[name] = {
                'mse': np.inf,
                'selected_features': []
            }

    return results


def main():
    np.random.seed(42)

    print("Starting linear simulation...")

    n_train = 70
    n_test = 1000
    n_features = 250
    n_simulations = 200

    print("Generating training and testing matrices...")
    X_train = np.random.randn(n_train, n_features)
    X_test = np.random.randn(n_test, n_features)

    X_train = (X_train - np.mean(X_train, axis=0)) / np.std(X_train, axis=0)
    X_test = (X_test - np.mean(X_test, axis=0)) / np.std(X_test, axis=0)

    print("Generating error matrix...")
    error_matrix = np.random.randn(n_train, n_simulations)

    os.makedirs('simulations/regression/linear/column_selections', exist_ok=True)
    os.makedirs('simulations/regression/linear/results', exist_ok=True)

    print("Generating column selections...")
    s_range = range(26)  # s = 0 to 25
    for s in s_range:
        if s == 0:
            selections_df = pd.DataFrame({'simulation': range(n_simulations), 'selected_columns': ['' for _ in range(n_simulations)]})
        else:
            selections = []
            for sim_idx in range(n_simulations):
                selected_cols = np.random.choice(n_features, size=s, replace=False)
                selections.append(','.join(map(str, selected_cols)))
            selections_df = pd.DataFrame({'simulation': range(n_simulations), 'selected_columns': selections})

        selections_df.to_csv(f'simulations/regression/linear/column_selections/s_{s}.csv', index=False)

    print("Running simulations...")

    for s in s_range:
        print(f"Processing s = {s}")

        selections_df = pd.read_csv(f'simulations/regression/linear/column_selections/s_{s}.csv')

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
                X_train, X_test, error_matrix[:, sim_idx],
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
                row[f'{method_name}_mse'] = method_result['mse']
                row[f'{method_name}_features'] = features_str
            results_data.append(row)

        results_df = pd.DataFrame(results_data)
        results_df.to_csv(f'simulations/regression/linear/results/s_{s}_results.csv', index=False)

    print("Simulation completed!")


if __name__ == "__main__":
    main()

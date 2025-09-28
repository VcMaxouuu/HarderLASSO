import numpy as np
import os
import sys
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from lassonet import LassoNetRegressorCV
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


class LassoNetFeatureSelector:
    """LASSO-Net with automatic feature selection through regularization path"""
    def __init__(self, hidden_dims=(20,), cv=5):
        self.hidden_dims = hidden_dims
        self.cv = cv
        self.selected_features_indices_ = []
        self.model = None

    def fit(self, X, y):
        self.model = LassoNetRegressorCV(
            hidden_dims=self.hidden_dims,
            cv=self.cv,
            verbose=0
        )
        self.model.fit(X, y)

        self.selected_features_indices_ = np.nonzero(self.model.best_selected_.numpy())[0].tolist()

    def predict(self, X):
        return self.model.predict(X)


def run_single_simulation(args):
    """Run a single simulation for given parameters"""
    X_train, X_test, error_col, selected_indices, s_value, sim_idx = args
    np.random.seed(42 + sim_idx)

    if s_value == 0:
        y_train = error_col
        y_test = np.zeros(X_test.shape[0])
    else:
        y_train = np.zeros(X_train.shape[0])
        y_test = np.zeros(X_test.shape[0])

        X_train_s = X_train[:, selected_indices]
        X_test_s = X_test[:, selected_indices]
        for i in range(0, len(selected_indices), 2):
            y_train += 10 * np.abs(X_train_s[:, i+1] - X_train_s[:, i])
            y_test += 10 * np.abs(X_test_s[:, i+1] - X_test_s[:, i])

        y_train += error_col

    # Initialize methods
    methods = {
        'RandomForest': RandomForestFeatureSelector(),
        'XGBoost': XGBoostFeatureSelector(),
        'LassoNet': LassoNetFeatureSelector(hidden_dims=(20,), cv=5),
        'LASSO_QUT': HarderLASSORegressor(hidden_dims=(20, ), penalty='l1'),
        'HarderLASSO_QUT': HarderLASSORegressor(hidden_dims=(20, ), penalty='harder')
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

    print("Starting nonlinear simulation...")

    n_train = 500
    n_test = 1000
    n_features = 50
    n_simulations = 200

    print("Generating training and testing matrices...")
    X_train = np.random.randn(n_train, n_features)
    X_test = np.random.randn(n_test, n_features)

    X_train = (X_train - np.mean(X_train, axis=0)) / np.std(X_train, axis=0)
    X_test = (X_test - np.mean(X_test, axis=0)) / np.std(X_test, axis=0)

    print("Generating error matrix...")
    error_matrix = np.random.randn(n_train, n_simulations)

    # Create directories for results
    os.makedirs('simulations/regression/nonlinear/column_selections', exist_ok=True)
    os.makedirs('simulations/regression/nonlinear/results', exist_ok=True)

    print("Generating column selections...")
    s_range = np.arange(0, 21, 2)
    for s in s_range:
        if s == 0:
            selections_df = pd.DataFrame({
                'simulation': range(n_simulations),
                'selected_columns': ['' for _ in range(n_simulations)]
            })
        else:
            selections = []
            for sim_idx in range(n_simulations):
                selected_cols = np.random.choice(n_features, size=s, replace=False)
                selections.append(','.join(map(str, selected_cols)))
            selections_df = pd.DataFrame({
                'simulation': range(n_simulations),
                'selected_columns': selections
            })

        selections_df.to_csv(f'simulations/regression/nonlinear/column_selections/s_{s}.csv', index=False)

    print("Running simulations...")

    for s in s_range:
        print(f"Processing s = {s}")

        selections_df = pd.read_csv(f'simulations/regression/nonlinear/column_selections/s_{s}.csv')

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
        results_df.to_csv(f'simulations/regression/nonlinear/results/s_{s}_results.csv', index=False)

    print("Nonlinear simulation completed!")


if __name__ == "__main__":
    main()

import numpy as np
import os
import sys
import pandas as pd
from sksurv.metrics import concordance_index_censored
from lassonet import LassoNetCoxRegressorCV
from sksurv.ensemble import GradientBoostingSurvivalAnalysis
from boruta import BorutaPy

sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
from HarderLASSO import HarderLASSOCox
import warnings
from multiprocessing import Pool

warnings.filterwarnings('ignore')


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
        t_train = 0
        t_test = 0
        for i in range(0, len(selected_indices), 2):
            t_train += 10 * np.abs(X_train_s[:, i+1] - X_train_s[:, i])
            t_test += 10 * np.abs(X_test_s[:, i+1] - X_test_s[:, i])

        t_train = np.exp(t_train)
        t_test = np.exp(t_test)

    T_train = np.random.exponential(scale=1/t_train)
    T_test = np.random.exponential(scale=1/t_test)

    C = np.quantile(T_train, 0.5)

    y_train = np.minimum(T_train, C)
    y_test = np.minimum(T_test, C)
    c_train = T_train <= C
    c_test = T_test <= C

    # Fit models
    methods = {
        'LassoNet': LassoNetFeatureSelector(hidden_dims=(20, ), cv=5),
        'GBoost': GBoostFeatureSelector(),
        'LASSO_QUT': HarderLASSOCox(penalty='l1', hidden_dims=(20, )),
        'HarderLASSO_QUT': HarderLASSOCox(penalty='harder', hidden_dims=(20,)),
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

    n_train = 750
    n_test = 1000
    n_features = 50
    n_simulations = 200

    print("Generating training and testing matrices...")
    X_train = np.random.randn(n_train, n_features)
    X_test = np.random.randn(n_test, n_features)

    X_train = (X_train - np.mean(X_train, axis=0)) / np.std(X_train, axis=0)
    X_test = (X_test - np.mean(X_test, axis=0)) / np.std(X_test, axis=0)

    os.makedirs('simulations/survival/nonlinear/column_selections', exist_ok=True)
    os.makedirs('simulations/survival/nonlinear/results', exist_ok=True)

    print("Generating column selections...")
    s_range = np.arange(0, 16, 2)
    for s in s_range:
        if s == 0:
            selections_df = pd.DataFrame({'simulation': range(n_simulations), 'selected_columns': ['' for _ in range(n_simulations)]})
        else:
            selections = []
            for sim_idx in range(n_simulations):
                selected_cols = np.random.choice(n_features, size=s, replace=False)
                selections.append(','.join(map(str, selected_cols)))
            selections_df = pd.DataFrame({'simulation': range(n_simulations), 'selected_columns': selections})

        selections_df.to_csv(f'simulations/survival/nonlinear/column_selections/s_{s}.csv', index=False)

    print("Running simulations...")

    for s in s_range:
        print(f"Processing s = {s}")

        selections_df = pd.read_csv(f'simulations/survival/nonlinear/column_selections/s_{s}.csv')

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
        results_df.to_csv(f'simulations/survival/nonlinear/results/s_{s}_results.csv', index=False)

    print("Simulation completed!")


if __name__ == "__main__":
    main()

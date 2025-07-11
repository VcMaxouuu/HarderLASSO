"""
Base HarderLASSO model implementation.

This module contains the core _BaseHarderLASSOModel class that provides
the fundamental functionality for neural network-based feature selection
using the HarderLASSO methodology.
"""

import torch
import numpy as np
from typing import Optional, Union, List, Tuple, Any, Dict
from ._network import _NeuralNetwork
from ._feature_selection import _FeatureSelectionMixin
from .._training import _FeatureSelectionTrainer, _FISTAOptimizer
from .._utils import _PhaseSpec


class _BaseHarderLASSOModel(_FeatureSelectionMixin):
    """Base class for HarderLASSO neural network models with feature selection.

    This class combines neural network functionality with feature selection
    capabilities using regularized optimization. It implements a multi-phase
    training approach with adaptive regularization.

    Parameters
    ----------
    hidden_dims : tuple of int, default=(20,)
        Number of neurons in each hidden layer.
    output_dim : int, default=1
        Number of output features. Must be positive integer.
    bias : bool, default=True
        Whether to include bias terms in the neural network layers.
    lambda_qut : float, optional
        Regularization parameter for controlling sparsity. If None, computed
        automatically using QUT methodology.
    penalty : {'lasso', 'harder', 'scad'}, default='harder'
        Type of penalty to apply.

    Attributes
    ----------
    _neural_network : _NeuralNetwork
        The underlying neural network architecture.
    lambda_qut_ : float
        Computed regularization parameter after fitting.
    selected_features_indices_ : list
        Indices of selected features after fitting.
    feature_names_in_ : list
        Names of input features from training data.
    _is_fitted : bool
        Whether the model has been fitted to data.

    Examples
    --------
    >>> model = _BaseHarderLASSOModel(
    ...     hidden_dims=(10, 5),
    ...     output_dim=1,
    ...     penalty='harder'
    ... )
    >>> # Note: This is an internal class - use public classes instead
    """

    def __init__(
        self,
        hidden_dims: Optional[Tuple[int, ...]] = (20,),
        output_dim: int = 1,
        bias: bool = True,
        lambda_qut: Optional[float] = None,
        penalty: str = 'harder'
    ):
        """Initialize the base HarderLASSO model.

        Parameters
        ----------
        hidden_dims : tuple of int, default=(20,)
            Number of neurons in each hidden layer.
        output_dim : int, default=1
            Number of output features.
        bias : bool, default=True
            Whether to include bias terms.
        lambda_qut : float, optional
            Regularization parameter for controlling sparsity.
        penalty : {'lasso', 'harder', 'scad'}, default='harder'
            Type of penalty to apply.
        """
        if penalty not in ['lasso', 'harder', 'scad']:
            raise ValueError(f"Penalty must be one of ['lasso', 'harder', 'scad'], got {penalty}")

        # Initialize neural network
        self._neural_network = _NeuralNetwork(hidden_dims, output_dim, bias)

        # Initialize feature selection mixin
        super().__init__(lambda_qut=lambda_qut, penalty=penalty)

        # Training state
        self._is_fitted = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the neural network.

        Applies feature selection (if trained) and passes input through
        the neural network layers.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (batch_size, n_features).

        Returns
        -------
        torch.Tensor
            Output tensor of shape (batch_size, output_dim).
        """
        if self.selected_features_indices_ is not None:
            x = x[:, self.selected_features_indices_]
        return self._neural_network.forward(x)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions using the trained model.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Input features. Can be numpy array or pandas DataFrame.

        Returns
        -------
        ndarray
            Predicted values of shape (n_samples, output_dim) or (n_samples,)
            for single output.

        Examples
        --------
        >>> predictions = model.predict(X_test)
        >>> print(predictions.shape)
        (100,)
        """
        if not self._is_fitted:
            raise ValueError("Model must be fitted before making predictions.")

        self._neural_network.eval()
        with torch.no_grad():
            if hasattr(self, 'preprocess_data'):
                X_tensor = self.preprocess_data(X, fit_scaler=False)
            else:
                X_tensor = torch.tensor(X, dtype=torch.float32)
            predictions = self.forward(X_tensor)

        if hasattr(self, 'format_predictions'):
            return self.format_predictions(predictions)

        return predictions.cpu().numpy()

    def _setup_model(self, input_dim: int) -> None:
        """Setup the model with the input dimension.

        This method is called automatically during fitting to initialize
        the neural network layers once the input dimension is known.

        Parameters
        ----------
        input_dim : int
            Number of input features.
        """
        if self._neural_network.layers is not None:
            return

        self._neural_network._build_layers(input_dim)
        self._neural_network._initialize_layers()
        super()._identify_parameter_types()

    def _create_training_phases(
        self,
        X_tensor: torch.Tensor,
        target_tensor: torch.Tensor,
        path_values: Optional[List[float]] = None,
        nu_path: Optional[List[float]] = None
    ) -> List[_PhaseSpec]:
        """Create training phases for the multi-phase optimization.

        This method defines the three-phase training strategy including
        initial Adam optimization, intermediate regularization phases,
        and final FISTA optimization for sparse solutions.

        Parameters
        ----------
        X_tensor : torch.Tensor
            Input features as tensor.
        target_tensor : torch.Tensor
            Target values as tensor.
        path_values : list of float, optional
            Custom regularization path. If None, computed automatically.
        nu_path : list of float, optional
            Custom nu path for harder penalty. If None, computed automatically.

        Returns
        -------
        list of _PhaseSpec
            List of phase specifications for training.
        """
        if not hasattr(self, 'QUT'):
            raise ValueError("QUT component required for lambda path computation.")

        lambda_path = self.QUT.compute_lambda_path(
            X_tensor,
            target_tensor,
            path_values=path_values,
            hidden_dims=self._neural_network.hidden_dims,
            activation=self._neural_network.activation
        )

        self.lambda_qut_ = self.QUT.lambda_qut_

        if self.penalty == 'harder' and nu_path is None:
            nu_path = [0.9, 0.7, 0.5, 0.3, 0.2, 0.1]

        # Build training phases
        phases = []
        all_parameters = lambda: self._neural_network.parameters()

        # Phase 1: Adam without regularization
        phases.append(
            _PhaseSpec(
                optimizer_class=torch.optim.Adam,
                parameter_getter=all_parameters,
                optimizer_kwargs={'lr': 0.1},
                regularization_kwargs={'lambda_': 0.0, 'penalty': self.penalty},
                relative_tolerance=1e-4,
                max_epochs=1000
            )
        )

        # Phase 2: Intermediate Adam phases with regularization
        for i, lambda_val in enumerate(lambda_path[:-1]):
            reg_kwargs = {'lambda_': lambda_val, 'penalty': self.penalty}
            if self.penalty == 'harder':
                reg_kwargs['nu'] = nu_path[i]
            elif self.penalty == 'scad':
                reg_kwargs['a'] = 3.7

            phases.append(
                _PhaseSpec(
                    optimizer_class=torch.optim.Adam,
                    parameter_getter=all_parameters,
                    optimizer_kwargs={'lr': 0.01},
                    regularization_kwargs=reg_kwargs,
                    relative_tolerance=1e-6
                )
            )

        # Phase 3: Final FISTA phase
        reg_kwargs = {'lambda_': lambda_path[-1], 'penalty': self.penalty}
        if self.penalty == 'harder':
            reg_kwargs['nu'] = nu_path[-1]
        elif self.penalty == 'scad':
            reg_kwargs['a'] = 3.7

        phases.append(
            _PhaseSpec(
                optimizer_class=_FISTAOptimizer,
                parameter_getter=lambda: self.penalized_parameters_,
                optimizer_kwargs={'lr': 0.01},
                regularization_kwargs=reg_kwargs,
                relative_tolerance=1e-10
            )
        )

        return phases


    def fit(
        self,
        X: Union[np.ndarray, Any],
        y: Union[np.ndarray, Any],
        verbose: bool = False,
        logging_interval: int = 50,
        lambda_path: Optional[List[float]] = None,
        nu_path: Optional[List[float]] = None
    ) -> '_BaseHarderLASSOModel':
        """Fit the HarderLASSO model to training data.

        This method implements the complete HarderLASSO training procedure
        including data preprocessing, neural network initialization, multi-phase
        optimization, and feature selection.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training features.
        y : array-like of shape (n_samples,) or (n_samples, n_outputs)
            Training targets.
        verbose : bool, default=False
            Whether to print training progress.
        logging_interval : int, default=50
            Frequency of progress updates during training.
        lambda_path : list of float, optional
            Custom regularization path. If None, computed automatically.
        nu_path : list of float, optional
            Custom nu path for harder penalty. If None, computed automatically.

        Returns
        -------
        self
            The fitted estimator.

        Examples
        --------
        >>> model.fit(X_train, y_train, verbose=True, logging_interval=100)
        >>> print(f"Selected {len(model.selected_features_)} features")
        """

        if (lambda_path is not None) != (nu_path is not None):
            raise ValueError("Both 'lambda_path' and 'nu_path' must be provided together.")

        if lambda_path is not None and nu_path is not None:
            if not (isinstance(lambda_path, list) and isinstance(nu_path, list)):
                raise TypeError("'lambda_path' and 'nu_path' must both be lists of floats.")

            if len(lambda_path) != len(nu_path):
                raise ValueError("'lambda_path' and 'nu_path' must be of the same length.")

            if not all(isinstance(nu, float) and 0 < nu < 1 for nu in nu_path):
                raise ValueError("'nu_path' must contain only floats strictly between 0 and 1.")

            if not all(isinstance(lam, float) for lam in lambda_path):
                raise ValueError("'lambda_path' must contain only floats.")


        # Store feature names
        self.feature_names_in_ = super()._extract_feature_names(X)

        # Preprocess data
        if hasattr(self, 'preprocess_data'):
            X_tensor, y_tensor = self.preprocess_data(X, y, fit_scaler=True)
        else:
            X_tensor = torch.tensor(X, dtype=torch.float32)
            y_tensor = torch.tensor(y, dtype=torch.float32)

        # Setup model architecture
        self._setup_model(X_tensor.shape[1])

        # Create training phases
        phases = self._create_training_phases(X_tensor, y_tensor, lambda_path, nu_path)

        # Train with multi-phase approach
        trainer = _FeatureSelectionTrainer(
            model=self,
            phases=phases,
            verbose=verbose,
            logging_interval=logging_interval
        )
        trainer.train(X_tensor, y_tensor)

        # Prune features
        super()._prune_features()

        # Final training without regularization
        self._train_without_regularization(
            X_tensor,
            y_tensor,
            verbose=verbose,
            logging_interval=logging_interval,
            relative_tolerance=1e-10,
            max_epochs=5000
        )

        self._is_fitted = True
        return self

    def _train_without_regularization(
        self,
        X: torch.Tensor,
        y: torch.Tensor,
        learning_rate: float = 0.01,
        verbose: bool = False,
        logging_interval: int = 50,
        relative_tolerance: float = 1e-5,
        max_epochs: int = 1000
    ) -> None:
        """Train the model without regularization after feature selection.

        This final training phase optimizes the model on selected features
        without any regularization to achieve the best possible fit.

        Parameters
        ----------
        X : torch.Tensor
            Input features tensor.
        y : torch.Tensor
            Target values tensor.
        learning_rate : float, default=0.01
            Learning rate for Adam optimizer.
        verbose : bool, default=False
            Whether to print training progress.
        logging_interval : int, default=50
            Frequency of progress updates.
        relative_tolerance : float, default=1e-5
            Convergence tolerance.
        max_epochs : int, default=1000
            Maximum number of training epochs.
        """
        from .._training import _ConvergenceChecker, _LoggingCallback

        logger = _LoggingCallback(logging_interval=logging_interval)
        optimizer = torch.optim.Adam(self._neural_network.parameters(), lr=learning_rate)

        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', patience=5, factor=0.5, min_lr=1e-10
        )

        # Training loop
        last_loss = torch.tensor(float('inf'), device=X.device)

        for epoch in range(max_epochs):
            optimizer.zero_grad()
            predictions = self.forward(X)
            loss = self.criterion(predictions, y)

            logger.log(epoch, loss, loss, verbose)

            # Check convergence
            if _ConvergenceChecker.check_convergence(loss, last_loss, relative_tolerance):
                if verbose:
                    print(f"Converged after {epoch} epochs. "
                          f"Relative loss change below {relative_tolerance}")
                break

            last_loss = loss
            loss.backward()
            optimizer.step()
            scheduler.step(loss)

    def summary(self) -> None:
        """
        Print a summary of the model architecture and selected features.

        This method provides an overview of:
        - Neural network architecture
        - Selected features and their names
        - Regularization parameters
        """
        print("### HarderLASSO Model Summary ###\n")
        self._neural_network.summary()

        print(f"\n=== Feature Selection Summary ===")
        if self._is_fitted:
            print(f"  - Input features: {len(self.feature_names_in_)}")
            print(f"  - Selected features: {len(self.selected_features_)}")
            print(f"  - Selection ratio: {len(self.selected_features_)/len(self.feature_names_in_):.2%}")
            print(f"  - Lambda (QUT): {self.lambda_qut_:.6f}")

            if len(self.selected_features_) <= 10:
                print(f"  - Selected feature names: {self.selected_features_}")
        else:
            print("Model not fitted yet.")

    def save(self, filepath: str) -> None:
        """
        Save the model to a file.

        Saves the complete model state including:
        - Neural network architecture and weights
        - Feature selection metadata
        - Preprocessing parameters
        - QUT computation results

        Args:
            filepath: Path where the model will be saved.

        Examples:
            >>> model.fit(X_train, y_train)
            >>> model.save('my_model.pth')
        """
        model_state = {
            'neural_network_state': self._neural_network.state_dict(),
            'selected_features_indices': self.selected_features_indices_,
            'hidden_dims': self._neural_network.hidden_dims,
            'feature_names_in': self.feature_names_in_,
            'lambda_qut': getattr(self, 'lambda_qut_', None),
            'penalty': self.penalty,
            'is_fitted': self._is_fitted
        }

        # Save preprocessing state if available
        if hasattr(self, '_scaler'):
            model_state['scaler'] = self._scaler

        # Save QUT information if available
        if hasattr(self, 'QUT') and self.QUT is not None:
            model_state['qut_info'] = {
                'lambda_path': getattr(self.QUT, 'lambda_path_', None),
                'monte_carlo_samples': getattr(self.QUT, 'monte_carlo_samples_', None),
                'alpha': self.QUT.alpha,
                'n_samples': self.QUT.n_samples
            }

        torch.save(model_state, filepath)

    @classmethod
    def load(cls, filepath: str) -> '_BaseHarderLASSOModel':
        """
        Load a saved model from file.

        Args:
            filepath: Path to the saved model file.

        Returns:
            The loaded model instance.

        Examples:
            >>> model = HarderLASSORegressor.load('my_model.pth')
            >>> predictions = model.predict(X_test)
        """
        model_state = torch.load(filepath, map_location='cpu')

        # Create model instance
        model = cls(
            hidden_dims=model_state['hidden_dims'],
            lambda_qut=model_state.get('lambda_qut'),
            penalty=model_state.get('penalty', 'harder')
        )

        # Restore state
        model.selected_features_indices_ = model_state['selected_features_indices']
        model.feature_names_in_ = model_state['feature_names_in']
        model._is_fitted = model_state.get('is_fitted', False)

        # Setup neural network
        if model.selected_features_indices_ is not None:
            input_dim = len(model.selected_features_indices_)
        else:
            input_dim = len(model.feature_names_in_)
        model._setup_model(input_dim)

        # Load neural network weights
        model._neural_network.load_state_dict(model_state['neural_network_state'])

        # Load preprocessing state
        if 'scaler' in model_state:
            model._scaler = model_state['scaler']

        # Load QUT information
        if 'qut_info' in model_state and hasattr(model, 'QUT'):
            qut_info = model_state['qut_info']
            model.QUT.lambda_path_ = qut_info.get('lambda_path')
            model.QUT.monte_carlo_samples_ = qut_info.get('monte_carlo_samples')
            model.QUT.alpha = qut_info.get('alpha', 0.05)
            model.QUT.n_samples = qut_info.get('n_samples', 5000)

        return model

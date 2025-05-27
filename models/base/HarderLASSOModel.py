import torch
from numpy import ndarray
from .BasicNN import BasicNN
from .FeatureSelectionMixin import FeatureSelectionMixin
from training import FeatureSelectionTrainer

class HarderLASSOModel(FeatureSelectionMixin):
    """
    A neural network model that follows the HarderLASSO approach by combining
    the functionality of `BasicNN` and `FeatureSelectionMixin`.

    Attributes:
        hidden_dims (tuple[int, ...]): A tuple specifying the number of neurons in each hidden layer.
        output_dim (int): The number of output features.
        bias (bool): Whether to include a bias term in the final layer.
        lambda_qut (float): Regularization parameter for controlling sparsity.
        nu (float): Regularization parameter for controlling the penalty.
    """
    def __init__(self,
                 hidden_dims: tuple[int, ...] | None,
                 output_dim: int,
                 bias: bool = True,
                 lambda_qut: float | None = None,
                 nu: float = 0.1):

        self.NN = BasicNN(hidden_dims, output_dim, bias)
        FeatureSelectionMixin.__init__(self, lambda_qut, nu)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Performs a forward pass through the network.

        Args:
            x (torch.Tensor): The input tensor.

        Returns:
            torch.Tensor: The output tensor after passing through the network.
        """
        if self.selected_features_idx_ is not None:
            x = x[:, self.selected_features_idx_]
        return self.NN.forward(x)

    def predict(self, X: ndarray) -> ndarray:
        """
        Makes predictions using the trained network.

        Args:
            X (np.ndarray): The input data as a NumPy array.

        Returns:
            np.ndarray: The predicted output as a NumPy array.
        """
        self.NN.eval()
        with torch.no_grad():
            if hasattr(self, 'preprocess_data'):
                X_tensor = self.preprocess_data(X, fit_scaler=False)
            else:
                X_tensor = torch.tensor(X, dtype=torch.float32)
            predictions = self.forward(X_tensor)

        if hasattr(self, 'format_predictions'):
            return self.format_predictions(predictions)

        return predictions.cpu().numpy()

    def _setup(self, input_dim: int) -> None:
        """
        Setup the model with the input dimension when it becomes available.

        Args:
            input_dim (int): Input dimension of the data.
        """
        if self.NN.layers is not None:
            return

        self.NN._build_layers(input_dim)
        self.NN._init_layers()
        super()._get_params_types()

    def fit(
            self,
            X,
            target,
            verbose=False,
            logging_interval=50,
            path_values: list[int] = None
        ):
        """
        Fits the model to the given data.

        Args:
            X (array-like): Input features.
            target (array-like): Target values.
            verbose (bool, optional): Whether to print training progress. Defaults to False.
            logging_interval (int, optional): Number of iterations between logging updates. Defaults to 50.
            path_values (list[float], optional): Custom lambda path values for regularization strength.
                If provided, this path will be used instead of computing a new one. Defaults to None.
        """
        self.in_features_ = super()._get_column_names(X)

        if hasattr(self, 'preprocess_data'):
            X_tensor, target_tensor = self.preprocess_data(X, target, fit_scaler=True)
        else:
            X_tensor = torch.tensor(X, dtype=torch.float32)
            target_tensor = torch.tensor(target, dtype=torch.float32)

        self._setup(X_tensor.shape[1])

        lambda_path, nu_path = self._compute_regularization_paths(X_tensor, target_tensor, path_values)

        trainer = FeatureSelectionTrainer(
            self,
            lambda_path=lambda_path,
            nu_path=nu_path,
            verbose=verbose,
            logging_interval=logging_interval
        )
        trainer.train(X_tensor, target_tensor)

        super()._prune()

        self._train_no_reg(
            X_tensor,
            target_tensor,
            verbose=verbose,
            logging_interval=logging_interval,
            rel_err=1e-10,
            max_epochs=5000
        )


    def _train_no_reg(
            self,
            X,
            target,
            learning_rate=0.01,
            verbose=False,
            logging_interval=50,
            rel_err=1e-5,
            max_epochs=1000,
        ):
        """
        Trains the model without any regularization.

        Args:
            X (array-like): Input features.
            target (array-like): Target values.
            learning_rate (float, optional): Learning rate for Adam optimizer. Defaults to 0.01.
            verbose (bool, optional): Whether to print training progress. Defaults to False.
            logging_interval (int, optional): Number of iterations between logging updates. Defaults to 50.
            rel_err (float, optional): Relative error for convergence. Defaults to 1e-5.
            max_epochs (int, optional): Maximum number of training epochs. Defaults to 1000.
        """
        from training import ConvergenceChecker, LoggingCallback

        logs = LoggingCallback(logging_interval=logging_interval)
        optimizer = torch.optim.Adam(self.NN.parameters(), lr=learning_rate)

        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', patience=5, factor=0.5, min_lr=1e-10
        )

        # Training loop
        epoch = 0
        last_loss = torch.tensor(float('inf'), device=X.device)

        for epoch in range(max_epochs):
            optimizer.zero_grad()
            predictions = self.forward(X)
            loss = self.criterion(predictions, target)

            logs.log(epoch, loss, loss, verbose)

            # Check for convergence
            if ConvergenceChecker.check_convergence(loss, last_loss, rel_err):
                if verbose:
                    print(f"Converged after {epoch} epochs. Relative penalized loss change below {rel_err}\n")
                break

            last_loss = loss
            loss.backward()
            optimizer.step()
            scheduler.step(loss)


    def summary(self):
        """
        Prints a summary of the model architecture and parameters.
        """
        self.NN.summary()
        print(f"Selected features: {self.selected_features}")
        print(f"Number of selected features: {len(self.selected_features)}")


    def save(self, path: str) -> None:
        """
        Save the model to a file.

        Saves both the model architecture and weights, along with preprocessing
        information and feature selection metadata.

        Args:
            path (str): Path where the model will be saved
        """
        model_state = {
            'nn_state_dict': self.NN.state_dict(),
            'selected_features_idx': self.selected_features_idx_,
            'hidden_dims': self.NN.hidden_dims,
            'input_features': self.in_features_,
            'lambda_qut': self.lambda_qut,
            'nu': self.nu,
            'scaler': self.scaler
        }

        # Save QUT information if available
        if hasattr(self, 'QUT') and self.QUT is not None:
            model_state['qut_info'] = {
                'lambda_path': self.QUT.lambda_path_,
                'monte_carlo_samples': self.QUT.monte_carlo_samples_,
                'alpha': self.QUT.alpha,
                'n_samples': self.QUT.n_samples
            }

        torch.save(model_state, path)

    @classmethod
    def load(cls, path: str):
        """
        Load a saved model.

        Args:
            path (str): Path to the saved model

        Returns:
            HarderLASSOModel: The loaded model
        """
        model_state = torch.load(path)

        # Create model
        model = cls(
            hidden_dims=model_state['hidden_dims'],
            lambda_qut=model_state['lambda_qut'],
            nu=model_state['nu']
        )

        if model.NN._output_dim is None:
            last_layer_key = None
            last_layer_idx = -1

            for key in model_state['nn_state_dict'].keys():
                if 'layers.' in key and '.weight' in key:
                    layer_idx = int(key.split('.')[1])
                    if layer_idx > last_layer_idx:
                        last_layer_idx = layer_idx
                        last_layer_key = key

            # Get output dimension from the last layer's weight shape
            model.NN._output_dim = model_state['nn_state_dict'][last_layer_key].shape[0]


        model.selected_features_idx_ = model_state['selected_features_idx']
        model.in_features_ = model_state['input_features']

        if model.selected_features_idx_ is not None:
            input_dim = len(model.selected_features_idx_)
        else:
            input_dim = len(model.in_features_)
        model._setup(input_dim)

        model.NN.load_state_dict(model_state['nn_state_dict'])
        model.scaler = model_state['scaler']

        # Load QUT information if available
        if 'qut_info' in model_state and hasattr(model, 'QUT'):
            model.QUT.lambda_path_ = model_state['qut_info']['lambda_path']
            model.QUT.monte_carlo_samples_ = model_state['qut_info']['monte_carlo_samples']
            model.QUT.alpha = model_state['qut_info']['alpha']
            model.QUT.n_samples = model_state['qut_info']['n_samples']

        return model

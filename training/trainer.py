import torch
from .FISTA import FISTA
from .proximal_operator import ProximalOperator
from .callbacks import ConvergenceChecker, LoggingCallback

class FeatureSelectionTrainer:
    """
    A trainer for feature selection models implementing a three-phase training approach:
    1. Initial phase: Adam optimizer for all parameters with no regularization
    2. Intermediate phases: Adam optimizer for all parameters with increasing regularization
    3. Final phase: FISTA with line search for penalized parameters only
    """
    def __init__(self, model, lambda_path=None, nu_path=None, verbose=False, logging_interval=50):
        self.model = model
        self.lambda_path = lambda_path
        self.nu_path = nu_path
        self.verbose = verbose
        self.logging_interval = logging_interval

        self.logging_callback = LoggingCallback(logging_interval=logging_interval)
        self.convergence_checker = ConvergenceChecker()

    def train(self, X, target):
        """
        Train the model through the three phases.

        Args:
            X (torch.Tensor): Training input data.
            target (torch.Tensor): Training target data.

        Returns:
            The trained model.
        """
        self.model.NN.train()

        lr = 0.1
        if self.verbose:
            print(f"### Phase 1/3: Adam with lambda = 0, lr={lr} ###")

        self._train_adam_phase(
            X=X,
            target=target,
            lambda_=0.0,
            nu=None,
            lr=lr,
            rel_err=1e-4,
            max_epochs=1000
        )

        if len(self.lambda_path) > 1:
            for i in range(len(self.lambda_path) - 1):
                lambda_ = self.lambda_path[i]
                nu = self.nu_path[i]
                lr = 0.01
                if self.verbose:
                    print(f"### Phase 2/3: Adam with lambda = {lambda_:.3f}, nu={nu}, lr={lr} ###")

                self._train_adam_phase(X, target, lambda_, nu, lr=lr, rel_err=1e-6)

        lambda_ = self.lambda_path[-1]
        nu = self.nu_path[-1]
        if self.verbose:
            print(f"### Phase 3/3: FISTA with lambda = {lambda_:.3f}, nu={nu} ###")

        self._train_fista_phase(X, target, lambda_, nu, lr=0.01, rel_err=1e-10)

        return self.model

    def _train_adam_phase(self, X, target, lambda_, nu, lr=0.01, rel_err=1e-5, max_epochs=None):
        optimizer = torch.optim.Adam(self.model.NN.parameters(), lr=lr)

        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', patience=5, factor=0.5, min_lr=1e-10
        )

        # Training loop
        epoch = 0
        last_loss = torch.tensor(float('inf'), device=X.device)


        while True:
            optimizer.zero_grad()
            predictions = self.model.forward(X)
            bare_loss = self.model.criterion(predictions, target)
            loss = bare_loss + self.model._regularization_term(lambda_, nu)

            self.logging_callback.log(epoch, loss, bare_loss, self.verbose)

            # Check for convergence
            if self.convergence_checker.check_convergence(loss, last_loss, rel_err):
                if self.verbose:
                    print(f"Converged after {epoch} epochs. Relative penalized loss change below {rel_err}\n")
                break

            if max_epochs is not None:
                if epoch > max_epochs:
                    if self.verbose:
                        print(f"\t Maximum number of epochs reached \n")
                    break

            last_loss = loss
            loss.backward()
            optimizer.step()
            scheduler.step(loss)
            epoch += 1

    def _train_fista_phase(self, X, target, lambda_, nu, lr=0.01, rel_err=1e-7):
        prox_opp = ProximalOperator(lambda_, nu)

        optimizer = FISTA(
            [
                {'params': self.model.penalized_parameters_, 'proximal_operator': prox_opp},
             ],
            lr = lr
        )

        # Define closure function
        def closure(backward=False):
            optimizer.zero_grad()
            predictions = self.model.forward(X)
            bare_loss = self.model.criterion(predictions, target)
            loss = bare_loss + self.model._regularization_term(lambda_, nu)
            if backward:
                bare_loss.backward()
            return loss, bare_loss

        # Training loop
        epoch = 0
        last_loss = torch.tensor(float('inf'), device=X.device)

        while True:
            loss, bare_loss = optimizer.step(closure)

            number_selected_features = None
            if self.verbose and epoch % self.logging_interval == 0:
                number_selected_features = self.model._count_nonzero_weights_layer0()[0]

            self.logging_callback.log(epoch, loss, bare_loss, self.verbose, number_selected_features)

            # Check for convergence
            if self.convergence_checker.check_convergence(loss, last_loss, rel_err):
                if self.verbose:
                    print(f"Converged after {epoch} epochs. Relative penalized loss change below {rel_err}\n")
                break


            last_loss = loss
            epoch += 1

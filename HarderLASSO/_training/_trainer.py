"""
Feature selection trainer for HarderLASSO models.

This module provides the _FeatureSelectionTrainer class that implements
the multi-phase training strategy for HarderLASSO models.
"""

import torch
from typing import List, Any, Optional
from ._optimizer import _ISTAOptimizer
from ._callbacks import _ConvergenceChecker, _LoggingCallback
from .._utils import HarderPenalty, SCADPenalty, MCPenalty

class _FeatureSelectionTrainer:
    """
    Trainer for feature selection models using dual optimizer strategy.

    This trainer implements the HarderLASSO training strategy with separate
    optimizers for penalized and non-penalized parameters:
    0. Initial phase: No regularization for warm start
    1. Non-penalized parameters: Single persistent Adam optimizer throughout training
    2. Penalized parameters: New Adam optimizer for each phase (except final)
    3. Final phase: ISTA for penalized parameters, Adam continues for non-penalized

    Parameters
    ----------
    model : object
        The HarderLASSO model to train.
    lambda_path : list of float
        List of regularization parameters for each phase.
    nu_path : list of float, optional
        List of nu parameters for HarderPenalty phases.
    verbose : bool, default=False
        Whether to print training progress.
    logging_interval : int, default=50
        Number of epochs between progress updates.

    Examples
    --------
    >>> trainer = _FeatureSelectionTrainer(model, lambda_path=[0.0, 0.1, 0.01], verbose=True)
    >>> trainer.train(X_tensor, y_tensor)
    """

    def __init__(
        self,
        model: Any,
        lambda_path: List[float],
        nu_path: Optional[List[float]] = None,
        verbose: bool = False,
        logging_interval: int = 50,
    ):
        """Initialize the feature selection trainer."""
        if lambda_path is None:
            raise ValueError("At least one lambda value must be specified.")

        self.model = model

        self.lambda_path = list(lambda_path)
        self.nu_path = nu_path or []

        self.verbose = verbose
        self.logging_interval = logging_interval

        # Initialize callbacks
        self.logger = _LoggingCallback(logging_interval)
        self.convergence_checker = _ConvergenceChecker()

        # Training state
        self.training_history = []

    def train(self, X: torch.Tensor, y: torch.Tensor) -> Any:
        """
        Train the model using the dual optimizer multi-phase approach.

        Parameters
        ----------
        X : torch.Tensor
            Input features tensor of shape (n_samples, n_features).
        y : torch.Tensor
            Target values tensor of shape (n_samples,) or (n_samples, n_outputs).

        Returns
        -------
        object
            The trained model.
        """
        # Set model to training mode
        self.model._neural_network.train()

        # Prepare nu_path for HarderPenalty if needed
        if isinstance(self.model.penalty, HarderPenalty) and not self.nu_path:
            raise ValueError(
                "nu_path must be provided for HarderPenalty. "
                "If not specified, use the default [0.9, 0.7, 0.5, 0.3, 0.2, 0.1]."
            )

        if self.verbose:
            phase_info = self._format_phase_info(phase_idx=-1, lambda_val=0, is_final_phase=False)
            self.logger.log_phase_start(f"No regularization phase", phase_info)

        # Run the phase
        phase_history = self._run_phase(X, y, phase_idx=-1, lambda_val=0, is_final_phase=False)
        self.training_history.append(phase_history)

        # Execute each training phase
        for phase_idx in range(len(self.lambda_path)):
            lambda_val = self.lambda_path[phase_idx]
            is_final_phase = (phase_idx == len(self.lambda_path)-1)

            if self.verbose:
                phase_info = self._format_phase_info(phase_idx=phase_idx, lambda_val=lambda_val, is_final_phase=is_final_phase)
                self.logger.log_phase_start(f"Phase {phase_idx+1}/{len(self.lambda_path)}", phase_info)

            # Run the phase
            phase_history = self._run_phase(X, y, phase_idx=phase_idx, lambda_val=lambda_val, is_final_phase=is_final_phase)
            self.training_history.append(phase_history)

        return self.model

    def _run_phase(
        self,
        X: torch.Tensor,
        y: torch.Tensor,
        phase_idx: int,
        lambda_val: float,
        is_final_phase: bool
    ) -> dict:
        """
        Run a single training phase with dual optimizer strategy.

        Parameters
        ----------
        X : torch.Tensor
            Input features tensor.
        y : torch.Tensor
            Target values tensor.
        phase_idx : int
            Phase index (0-based).
        lambda_val : float
            Regularization parameter for this phase.
        is_final_phase : bool
            Whether this is the final phase.

        Returns
        -------
        dict
            Dictionary with phase training history.
        """
        penalized_params = list(self.model.penalized_parameters_) if hasattr(self.model, 'penalized_parameters_') else []
        unpenalized_params = list(self.model.unpenalized_parameters_) if hasattr(self.model, 'unpenalized_parameters_') else []

        # Update penalty parameters for this phase
        penalty_updates = {'lambda_': lambda_val}
        if isinstance(self.model.penalty, HarderPenalty) and phase_idx < len(self.nu_path):
            penalty_updates['nu'] = self.nu_path[phase_idx]
        elif isinstance(self.model.penalty, SCADPenalty):
            penalty_updates['a'] = 3.7
        elif isinstance(self.model.penalty, MCPenalty):
            penalty_updates['gamma'] = 3

        # Update the penalty object with new parameters
        self.model.penalty.update_params(**penalty_updates)

        scheduler = None

        if is_final_phase:
            if not penalized_params:
                raise ValueError(f"No penalized parameters found for ISTA phase {phase_idx + 1}")

            ISTA_optimizer = _ISTAOptimizer([
                {'params': penalized_params, 'penalty': self.model.penalty},
                {'params': unpenalized_params}
            ], lr=0.01, lambda_=lambda_val)

        else:
            optimizer = torch.optim.Adam(
                    unpenalized_params + penalized_params,
                    lr=0.01
                )

            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                    optimizer=optimizer, mode='min', patience=5, factor=0.5, min_lr=1e-10
                )

        # Set phase-specific tolerances and max epochs
        if phase_idx == -1:
            relative_tolerance = 1e-4
            max_epochs = 1000
        elif is_final_phase:
            relative_tolerance = 1e-10
            max_epochs = None
        else:
            relative_tolerance = 1e-6
            max_epochs = None

        # Training loop
        epoch = 0
        last_loss = torch.tensor(float("inf"), device=X.device)

        def closure(backward: bool = False) -> tuple:
            """Closure function for ISTA optimizer."""
            ISTA_optimizer.zero_grad()

            predictions = self.model.forward(X)
            bare_loss = self.model.criterion(predictions, y)
            penalty = self.model.penalty.value(
                parameter=self.model.penalized_parameters_[0]
            )
            total_loss = bare_loss + penalty

            if backward:
                bare_loss.backward()

            return total_loss, bare_loss

        while True:
            if is_final_phase and isinstance(ISTA_optimizer, _ISTAOptimizer):
                total_loss, bare_loss = ISTA_optimizer.step(closure)
            else:
                optimizer.zero_grad()

                predictions = self.model.forward(X)
                bare_loss = self.model.criterion(predictions, y)
                penalty = self.model.penalty.value(
                    parameter=self.model.penalized_parameters_[0]
                )
                total_loss = bare_loss + penalty
                total_loss.backward()

                optimizer.step()

            if self.convergence_checker.check_convergence(total_loss, last_loss, relative_tolerance):
                if self.verbose:
                    self.logger.log_convergence(epoch, relative_tolerance)
                break

            # Log progress
            current_n_features, _ = self.model._count_nonzero_weights_first_layer()
            self.logger.log(
                epoch,
                total_loss.detach(),
                bare_loss.detach(),
                self.verbose,
                n_features=current_n_features,
                phase_name=f"Phase {phase_idx + 1}"
            )

            # Check maximum epochs
            if max_epochs and epoch >= max_epochs:
                if self.verbose:
                    self.logger.log_early_stopping(epoch, "Maximum epochs reached")
                break

            # Update for next iteration
            last_loss = total_loss

            # Update learning rate schedulers
            if scheduler:
                scheduler.step(total_loss)

            epoch += 1

        # Return phase history
        return {"epochs": epoch, "final_loss": total_loss.item()}


    def _format_phase_info(self, phase_idx: int, lambda_val: float, is_final_phase: bool) -> str:
        """
        Format phase information for logging.

        Parameters
        ----------
        phase_idx : int
            Phase index (0-based).
        lambda_val : float
            Regularization parameter value.
        is_final_phase : bool
            Whether this is the final phase.

        Returns
        -------
        str
            Formatted phase information string.
        """
        if is_final_phase:
            optimizer_name = "ISTA (penalized)"
        else:
            optimizer_name = "Adam (both parameter groups)"

        # Format regularization parameters
        if lambda_val == 0.0:
            reg_info = "no regularization"
        else:
            reg_info = f"lambda={lambda_val}"
            if isinstance(self.model.penalty, HarderPenalty) and phase_idx < len(self.nu_path):
                reg_info += f", nu={self.nu_path[phase_idx]}"
            elif isinstance(self.model.penalty, SCADPenalty):
                reg_info += ", a=3.7"
            elif isinstance(self.model.penalty, MCPenalty):
                reg_info += ", gamma=3"

        return f"{optimizer_name}, {reg_info}"

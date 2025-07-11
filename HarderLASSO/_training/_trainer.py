"""
Feature selection trainer for HarderLASSO models.

This module provides the _FeatureSelectionTrainer class that implements
the multi-phase training strategy for HarderLASSO models.
"""

import torch
from typing import List, Any, Callable, Optional
from ._optimizer import _FISTAOptimizer
from ._callbacks import _ConvergenceChecker, _LoggingCallback
from .._utils import _PhaseSpec


class _FeatureSelectionTrainer:
    """
    Trainer for feature selection models using multi-phase optimization.

    This trainer implements the HarderLASSO training strategy:
    1. Initial phase: Adam optimizer with no regularization
    2. Intermediate phases: Adam optimizer with increasing regularization
    3. Final phase: FISTA optimizer for sparse solution

    Parameters
    ----------
    model : object
        The HarderLASSO model to train.
    phases : list of _PhaseSpec
        List of training phase specifications.
    verbose : bool, default=False
        Whether to print training progress.
    logging_interval : int, default=50
        Number of epochs between progress updates.

    Examples
    --------
    >>> phases = [
    ...     _PhaseSpec(torch.optim.Adam, lambda: model.parameters(), {'lr': 0.1}),
    ...     _PhaseSpec(_FISTAOptimizer, lambda: model._penalized_parameters, {'lr': 0.01})
    ... ]
    >>> trainer = _FeatureSelectionTrainer(model, phases, verbose=True)
    >>> trainer.train(X_tensor, y_tensor)
    """

    def __init__(
        self,
        model: Any,
        phases: List[_PhaseSpec],
        verbose: bool = False,
        logging_interval: int = 50
    ):
        """Initialize the feature selection trainer."""
        if not phases:
            raise ValueError("At least one training phase must be specified.")

        self.model = model
        self.phases = list(phases)
        self.verbose = verbose
        self.logging_interval = logging_interval

        # Initialize callbacks
        self.logger = _LoggingCallback(logging_interval)
        self.convergence_checker = _ConvergenceChecker()

        # Training state
        self.current_phase = 0
        self.training_history = []

    def train(self, X: torch.Tensor, y: torch.Tensor) -> Any:
        """
        Train the model using the multi-phase approach.

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

        # Execute each training phase
        for phase_idx, phase in enumerate(self.phases, 1):
            if self.verbose:
                phase_info = self._format_phase_info(phase)
                self.logger.log_phase_start(f"Phase {phase_idx}/{len(self.phases)}", phase_info)

            # Run the phase
            phase_history = self._run_phase(X, y, phase, phase_idx)
            self.training_history.append(phase_history)

            if self.verbose:
                final_loss = phase_history['final_loss']
                epochs = phase_history['epochs']
                self.logger.log_phase_end(f"Phase {phase_idx}", final_loss, epochs)

        return self.model

    def _run_phase(
        self,
        X: torch.Tensor,
        y: torch.Tensor,
        phase: _PhaseSpec,
        phase_idx: int
    ) -> dict:
        """
        Run a single training phase.

        Parameters
        ----------
        X : torch.Tensor
            Input features tensor.
        y : torch.Tensor
            Target values tensor.
        phase : _PhaseSpec
            Phase specification.
        phase_idx : int
            Phase index for logging.

        Returns
        -------
        dict
            Dictionary with phase training history.
        """
        params = list(phase.parameter_getter())
        if not params:
            raise ValueError(f"No parameters to optimize in phase {phase_idx}")

        optimizer = self._create_optimizer(phase, params)

        scheduler = None
        if not isinstance(optimizer, _FISTAOptimizer):
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, mode='min', patience=5, factor=0.5, min_lr=1e-10
            )

        # Training loop
        epoch = 0
        last_loss = torch.tensor(float("inf"), device=X.device)
        phase_history = {
            'phase_idx': phase_idx,
            'losses': [],
            'bare_losses': [],
            'learning_rates': [],
            'epochs': 0,
            'converged': False,
            'final_loss': None
        }

        def closure(backward: bool = False) -> tuple:
            optimizer.zero_grad()

            predictions = self.model.forward(X)
            bare_loss = self.model.criterion(predictions, y)
            reg_loss = self.model._compute_regularization_term(**phase.regularization_kwargs)
            total_loss = bare_loss + reg_loss

            if backward:
                bare_loss.backward()

            return total_loss, bare_loss

        while True:
            if isinstance(optimizer, _FISTAOptimizer):
                total_loss, bare_loss = optimizer.step(closure)
            else:
                optimizer.zero_grad()

                predictions = self.model.forward(X)
                bare_loss = self.model.criterion(predictions, y)
                reg_loss = self.model._compute_regularization_term(**phase.regularization_kwargs)
                total_loss = bare_loss + reg_loss

                total_loss.backward()
                optimizer.step()

            # Log progress
            current_n_features = self._count_selected_features()
            self.logger.log(
                epoch,
                total_loss.detach(),
                bare_loss.detach(),
                self.verbose,
                n_features=current_n_features,
                phase_name=f"Phase {phase_idx}"
            )

            # Optional : Record history
            #   phase_history['losses'].append(total_loss.item())
            #   phase_history['bare_losses'].append(bare_loss.item())
            #   if hasattr(optimizer, 'get_lr'):
            #       phase_history['learning_rates'].append(optimizer.get_lr()[0])

            # Check convergence
            if self.convergence_checker.check_convergence(
                total_loss, last_loss, phase.relative_tolerance
            ):
                if self.verbose:
                    self.logger.log_convergence(epoch, phase.relative_tolerance)
                phase_history['converged'] = True
                break

            # Check maximum epochs
            if phase.max_epochs and epoch >= phase.max_epochs:
                if self.verbose:
                    self.logger.log_early_stopping(epoch, "Maximum epochs reached")
                break

            # Update for next iteration
            last_loss = total_loss
            if scheduler:
                scheduler.step(total_loss)
            epoch += 1

        # Optional - Update history
        phase_history['epochs'] = epoch
        phase_history['final_loss'] = total_loss

        return phase_history

    def _create_optimizer(self, phase: _PhaseSpec, params: List[torch.Tensor]) -> torch.optim.Optimizer:
        """
        Create optimizer for a training phase.

        Parameters
        ----------
        phase : _PhaseSpec
            Phase specification.
        params : list of torch.Tensor
            Parameters to optimize.

        Returns
        -------
        torch.optim.Optimizer
            Configured optimizer.
        """
        if issubclass(phase.optimizer_class, _FISTAOptimizer):
            # FISTA needs proximal operator
            from . import _create_proximal_operator
            prox_op = _create_proximal_operator(**phase.regularization_kwargs)

            # Create parameter groups with proximal operator
            optimizer = phase.optimizer_class(params, prox_op, **phase.optimizer_kwargs)
        else:
            # Standard optimizer
            optimizer = phase.optimizer_class(params, **phase.optimizer_kwargs)

        return optimizer

    def _format_phase_info(self, phase: _PhaseSpec) -> str:
        """
        Format phase information for logging.

        Parameters
        ----------
        phase : _PhaseSpec
            Phase specification.

        Returns
        -------
        str
            Formatted phase information string.
        """
        optimizer_name = phase.optimizer_class.__name__

        # Format regularization parameters
        if phase.regularization_kwargs:
            reg_info = ", ".join(f"{k}={v}" for k, v in phase.regularization_kwargs.items())
        else:
            reg_info = "no regularization"

        return f"{optimizer_name} optimizer, {reg_info}"

    def _count_selected_features(self) -> Optional[int]:
        """
        Count the number of currently selected features.

        Returns
        -------
        int or None
            Number of selected features, or None if not available.
        """
        try:
            if hasattr(self.model, '_count_nonzero_weights_first_layer'):
                count, _ = self.model._count_nonzero_weights_first_layer()
                return count
        except:
            pass
        return None

    def get_training_history(self) -> List[dict]:
        """
        Get the complete training history.

        Returns
        -------
        list of dict
            List of dictionaries containing training history for each phase.
        """
        return self.training_history

    def reset(self) -> None:
        """Reset the trainer state."""
        self.current_phase = 0
        self.training_history = []

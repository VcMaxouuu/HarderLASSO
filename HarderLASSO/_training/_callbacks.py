"""
Training callbacks for HarderLASSO models.

This module provides callback functions for monitoring and controlling
the training process.
"""

import torch
from typing import Optional, Any


class _ConvergenceChecker:
    """
    Utility class for checking training convergence.

    This class provides methods to determine if the training process
    has converged based on relative loss changes.
    """

    @staticmethod
    def check_convergence(
        current_loss: torch.Tensor,
        previous_loss: torch.Tensor,
        relative_tolerance: float
    ) -> bool:
        """
        Check if training has converged based on relative loss change.

        Parameters
        ----------
        current_loss : torch.Tensor
            Current epoch loss value.
        previous_loss : torch.Tensor
            Previous epoch loss value.
        relative_tolerance : float
            Relative tolerance for convergence.

        Returns
        -------
        bool
            True if converged, False otherwise.
        """
        if torch.isinf(previous_loss) or torch.isnan(previous_loss):
            return False

        if torch.isinf(current_loss) or torch.isnan(current_loss):
            return False

        # Compute relative change
        if torch.abs(current_loss) < 1e-12:
            return True

        relative_change = torch.abs(current_loss - previous_loss) / torch.abs(previous_loss)
        return relative_change < relative_tolerance

    @staticmethod
    def check_improvement(
        current_loss: torch.Tensor,
        best_loss: torch.Tensor,
        min_improvement: float = 1e-6
    ) -> bool:
        """
        Check if current loss is an improvement over best loss.

        Parameters
        ----------
        current_loss : torch.Tensor
            Current loss value.
        best_loss : torch.Tensor
            Best loss seen so far.
        min_improvement : float, default=1e-6
            Minimum improvement required.

        Returns
        -------
        bool
            True if current loss is better than best loss by at least min_improvement.
        """
        if torch.isinf(best_loss) or torch.isnan(best_loss):
            return True

        if torch.isinf(current_loss) or torch.isnan(current_loss):
            return False

        return current_loss < best_loss - min_improvement


class _LoggingCallback:
    """
    Callback for logging training progress.

    This callback prints training metrics at specified intervals
    to monitor the training process.

    Parameters
    ----------
    logging_interval : int
        Number of epochs between log messages.

    Examples
    --------
    >>> logger = _LoggingCallback(logging_interval=10)
    >>> logger.log(epoch=50, loss=0.123, bare_loss=0.098, verbose=True)
    """

    def __init__(self, logging_interval: int = 50):
        """Initialize the logging callback."""
        if logging_interval <= 0:
            raise ValueError("Logging interval must be positive.")

        self.logging_interval = logging_interval
        self.epoch_count = 0

    def log(
        self,
        epoch: int,
        loss: torch.Tensor,
        bare_loss: torch.Tensor,
        verbose: bool,
        n_features: Optional[int] = None,
        phase_name: Optional[str] = None
    ) -> None:
        """
        Log training progress.

        Parameters
        ----------
        epoch : int
            Current epoch number.
        loss : torch.Tensor
            Total loss including regularization.
        bare_loss : torch.Tensor
            Loss without regularization.
        verbose : bool
            Whether to actually print the log.
        n_features : int, optional
            Number of selected features (if available).
        phase_name : str, optional
            Name of the current training phase.
        """
        if not verbose:
            return

        if epoch % self.logging_interval == 0:
            # Format the loss values
            loss_str = f"{loss.item():.6f}"
            bare_loss_str = f"{bare_loss.item():.6f}"

            # Create base message
            msg = f"  Epoch {epoch:4d}: Loss = {loss_str}, Bare Loss = {bare_loss_str}"

            # Add feature count if available
            if n_features is not None:
                msg += f", Features = {n_features}"

            print(msg)

    def log_phase_start(self, phase_name: str, phase_info: str) -> None:
        """
        Log the start of a new training phase.

        Parameters
        ----------
        phase_name : str
            Name of the phase.
        phase_info : str
            Additional information about the phase.
        """
        print(f"\n=== Starting {phase_name} ===")
        if phase_info:
            print(f"\t{phase_info}")

    def log_phase_end(self, phase_name: str, final_loss: torch.Tensor, epochs: int) -> None:
        """
        Log the end of a training phase.

        Parameters
        ----------
        phase_name : str
            Name of the phase.
        final_loss : torch.Tensor
            Final loss value.
        epochs : int
            Number of epochs completed.
        """
        print(f"=== Completed {phase_name}:  Final loss: {final_loss.item():.6f} - Epochs: {epochs} ===")

    def log_convergence(self, epoch: int, tolerance: float) -> None:
        """
        Log convergence information.

        Parameters
        ----------
        epoch : int
            Epoch at which convergence was achieved.
        tolerance : float
            Tolerance used for convergence check.
        """
        print(f"\tConverged at epoch {epoch} (tolerance: {tolerance:.2e})")

    def log_early_stopping(self, epoch: int, reason: str) -> None:
        """
        Log early stopping information.

        Parameters
        ----------
        epoch : int
            Epoch at which training was stopped.
        reason : str
            Reason for early stopping.
        """
        print(f"\tEarly stopping at epoch {epoch}: {reason}")

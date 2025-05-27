import torch

class ConvergenceChecker:
    @staticmethod
    def check_convergence(loss, last_loss, rel_err):
        """
        Checks if the training has converged based on relative error.
        """
        if torch.isinf(last_loss):
            return False
        relative_change = torch.abs(loss - last_loss) / torch.abs(last_loss)
        bound_reached = relative_change < rel_err
        return bound_reached

class LoggingCallback:
    def __init__(self, logging_interval=50):
        """
        Initializes a logging callback.

        Parameters
        ----------
        logging_interval : int
            The interval (in epochs) at which logs are printed.
        """
        self.logging_interval = logging_interval

    def log(self, epoch, loss, bare_loss, verbose, n_feat=None):
        """
        Logs training progress.
        """
        if verbose and epoch % self.logging_interval == 0:
            msg = f"\t Epoch {epoch}: Penalized Loss = {loss.item():.5f}, Bare Loss = {bare_loss.item():.5f}"
            if n_feat is not None:
                msg += f", Number of Features = {n_feat}"
            print(msg)

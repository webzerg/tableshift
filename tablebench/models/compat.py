import abc
from abc import ABC, abstractmethod


class SklearnStylePytorchModel(ABC, nn.Module):
    """A pytorch model with an sklearn-style interface."""

    def __init__(self):
        super().__init__()

    def init_layers(self):
        raise

    @abstractmethod
    def forward(self, X):
        raise

    @abstractmethod
    def predict(self, X) -> np.ndarray:
        raise

    @abstractmethod
    def predict_proba(self, X) -> np.ndarray:
        raise

    def print_summary(self, global_step: int, metrics):
        # Print a summary every n steps
        if (global_step % 100) == 0:
            metrics_str = ', '.join(
                [f"{k}: {v}" for k, v in sorted(metrics.items())])
            logging.info(
                "metrics for model {} at step {}: {}".format(
                    self.model_type, global_step, metrics_str))

    def disparity_metric_fn(self) -> Callable:
        raise

    def _check_inputs(self, X, y):
        raise

    def _compute_validation_metrics(self, X_val,
                                    y_val, sens_val) -> dict:
        raise

    def _update(self, optimizer, X, y, g):
        """Execute a single parameter update step."""
        raise

    @abstractmethod
    def fit(self, **kwargs):
        raise

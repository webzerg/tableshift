from abc import ABC, abstractmethod
import numpy as np
from torch import nn


class SklearnStylePytorchModel(ABC, nn.Module):
    """A pytorch model with an sklearn-style interface."""

    def __init__(self):
        super().__init__()

    @abstractmethod
    def predict(self, X) -> np.ndarray:
        raise

    @abstractmethod
    def predict_proba(self, X) -> np.ndarray:
        raise

    @abstractmethod
    def fit(self, **kwargs):
        raise

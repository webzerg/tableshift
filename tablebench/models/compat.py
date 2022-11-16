from abc import ABC, abstractmethod

import numpy as np
from torch import nn


class SklearnStylePytorchModel(ABC, nn.Module):
    """A pytorch model with an sklearn-style interface."""

    def __init__(self):
        super().__init__()

    def predict(self, X) -> np.ndarray:
        """sklearn-compatible prediction function."""
        return self(X).detach().cpu().numpy()

    @abstractmethod
    def predict_proba(self, X) -> np.ndarray:
        """sklearn-compatible probability prediction function."""
        raise


    @abstractmethod
    def fit(self, **kwargs):
        raise

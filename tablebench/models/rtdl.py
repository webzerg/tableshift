"""
Wrappers for tabular baseline models from the rtdl package.

rtdl source: https://github.com/Yura52/rtdl
"""

from typing import Mapping, Optional, Callable

import numpy as np
import rtdl
import scipy
import torch

from tablebench.models.compat import SklearnStylePytorchModel
from tablebench.models.training import train_epoch


@torch.no_grad()
def predict_proba(model, X):
    prediction = model.predict(X)
    return scipy.special.expit(prediction)


class SklearnStyleRTDLModel(SklearnStylePytorchModel):

    def train_epoch(self, train_loader: torch.utils.data.DataLoader,
                    optimizer: torch.optim.Optimizer,
                    loss_fn: Callable,
                    other_loaders: Optional[
                        Mapping[str, torch.utils.data.DataLoader]] = None
                    ):
        """Run a single epoch of model training."""
        train_epoch(self, optimizer, loss_fn, train_loader)

    def predict_proba(self, X) -> np.ndarray:
        raise


class ResNetModel(rtdl.ResNet, SklearnStyleRTDLModel):

    def predict_proba(self, X) -> np.ndarray:
        return predict_proba(self, X)


class MLPModel(rtdl.MLP, SklearnStyleRTDLModel):
    def predict_proba(self, X) -> np.ndarray:
        return predict_proba(self, X)


class FTTransformerModel(rtdl.FTTransformer, SklearnStyleRTDLModel):
    def predict_proba(self, X) -> np.ndarray:
        return predict_proba(self, X)

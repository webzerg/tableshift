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
from tablebench.models.utils import apply_model


@torch.no_grad()
def predict_proba(model, X):
    prediction = model.predict(X)
    return scipy.special.expit(prediction)


class SklearnStyleRTDLModel(SklearnStylePytorchModel):

    def __init__(self, device, **kwargs):
        super().__init__(**kwargs)
        self.device_ = torch.device(device)
        self.to(self.device_)

    def train_epoch(self, train_loader: torch.utils.data.DataLoader,
                    optimizer: torch.optim.Optimizer,
                    loss_fn: Callable,
                    other_loaders: Optional[
                        Mapping[str, torch.utils.data.DataLoader]] = None
                    ):
        """Run a single epoch of model training."""
        for iteration, (x_batch, y_batch, _) in enumerate(train_loader):
            self.train()
            optimizer.zero_grad()
            # TODO(jpgard): handle categorical features here.
            loss = loss_fn(apply_model(self, x_batch).squeeze(1), y_batch)
            loss.backward()
            optimizer.step()


class ResNetModel(rtdl.ResNet, SklearnStyleRTDLModel):

    def predict_proba(self, X) -> np.ndarray:
        return predict_proba(self, X)


class MLPModel(rtdl.MLP, SklearnStyleRTDLModel):
    def predict_proba(self, X) -> np.ndarray:
        return predict_proba(self, X)


class FTTransformerModel(rtdl.FTTransformer, SklearnStyleRTDLModel):
    def predict_proba(self, X) -> np.ndarray:
        return predict_proba(self, X)

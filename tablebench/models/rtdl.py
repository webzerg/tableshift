"""
Wrappers for tabular baseline models from the rtdl package.

rtdl source: https://github.com/Yura52/rtdl
"""

import copy
from typing import Mapping, Optional, Callable, Any

import numpy as np
import rtdl
import scipy
import torch

from tablebench.models.compat import SklearnStylePytorchModel, OPTIMIZER_ARGS
from tablebench.models.training import train_epoch


class SklearnStyleRTDLModel(SklearnStylePytorchModel):

    def train_epoch(self, train_loaders: torch.utils.data.DataLoader,
                    loss_fn: Callable,
                    device: str,
                    eval_loaders: Optional[
                        Mapping[str, torch.utils.data.DataLoader]] = None,
                    ) -> float:
        """Run a single epoch of model training."""
        return train_epoch(self, self.optimizer,
                           loss_fn, train_loaders, device=device)

    @torch.no_grad()
    def predict_proba(self, X):
        prediction = self.predict(X)
        return scipy.special.expit(prediction)


class ResNetModel(rtdl.ResNet, SklearnStyleRTDLModel):
    def __init__(self, **hparams):
        self.config = copy.deepcopy(hparams)

        # Remove hparams that are not taken by the rtdl constructor.
        for k in OPTIMIZER_ARGS:
            hparams.pop(k)

        super().__init__(**hparams)
        self._init_optimizer()

    def predict_proba(self, X) -> np.ndarray:
        return self.predict_proba(X)


class MLPModel(rtdl.MLP, SklearnStyleRTDLModel):
    def __init__(self, **hparams):
        self.config = copy.deepcopy(hparams)

        # Remove hparams that are not taken by the rtdl constructor.
        for k in OPTIMIZER_ARGS:
            hparams.pop(k)

        super().__init__(**hparams)
        self._init_optimizer()

    def predict_proba(self, X) -> np.ndarray:
        return self.predict_proba(X)


class FTTransformerModel(rtdl.FTTransformer, SklearnStyleRTDLModel):

    def predict_proba(self, X) -> np.ndarray:
        return self.predict_proba(X)

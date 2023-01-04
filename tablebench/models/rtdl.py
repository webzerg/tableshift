"""
Wrappers for tabular baseline models from the rtdl package.

rtdl source: https://github.com/Yura52/rtdl
"""

import copy
from typing import Optional, Callable, Any, Dict

import numpy as np
import rtdl
import scipy
import torch
from torch.utils.data import DataLoader

from tablebench.models.compat import SklearnStylePytorchModel, OPTIMIZER_ARGS
from tablebench.models.training import train_epoch


class SklearnStyleRTDLModel(SklearnStylePytorchModel):

    def train_epoch(self,
                    train_loaders: Dict[Any, DataLoader],
                    loss_fn: Callable,
                    device: str,
                    uda_loader: Optional[DataLoader] = None,
                    eval_loaders: Optional[Dict[str, DataLoader]] = None,
                    max_examples_per_epoch: Optional[int] = None
                    ) -> float:
        """Run a single epoch of model training."""
        assert len(train_loaders.values()) == 1
        train_loader = list(train_loaders.values())[0]
        return train_epoch(self, self.optimizer, loss_fn, train_loader, device)

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

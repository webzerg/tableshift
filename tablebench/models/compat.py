from abc import ABC, abstractmethod
from collections import defaultdict
from typing import Optional, Mapping, Union

import numpy as np
from ray import tune
import torch
from torch import nn


def append_by_key(from_dict: dict, to_dict: Union[dict, defaultdict]) -> dict:
    for k, v in from_dict.items():
        assert (k in to_dict) or (isinstance(to_dict, defaultdict))
        to_dict[k].append(v)
    return to_dict


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
    def train_epoch(self, **kwargs):
        """Conduct one epoch of training."""
        raise

    def fit(self,
            train_loader: torch.utils.data.DataLoader,
            optimizer: torch.optim.Optimizer,
            loss_fn,
            n_epochs=1,
            other_loaders: Optional[
                Mapping[str, torch.utils.data.DataLoader]] = None,
            tune_report_split=None) -> dict:
        fit_metrics = defaultdict(list)

        if tune_report_split:
            assert tune_report_split in list(other_loaders.keys()) + ["train"]

        for epoch in range(1, n_epochs + 1):
            metrics = self.train_epoch(train_loader=train_loader,
                                       optimizer=optimizer,
                                       loss_fn=loss_fn,
                                       other_loaders=other_loaders)
            log_str = f'Epoch {epoch:03d} ' + ' | '.join(
                f"{k} score: {v:.4f}" for k, v in metrics.items())
            print(log_str)

            if tune_report_split:
                # TODO(jpgard): consider reporting multiple named metrics here.
                #  e.g. tune.report(mean_acc=x); tune.report(wg_acc=y) etc.
                tune.report(metrics[tune_report_split])

            fit_metrics = append_by_key(from_dict=metrics, to_dict=fit_metrics)

        return fit_metrics

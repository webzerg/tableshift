from abc import ABC, abstractmethod
from collections import defaultdict
from typing import Optional, Mapping, Union, Callable, Any

import numpy as np
import os

from ray.air import session
from ray.air.checkpoint import Checkpoint
import torch
from torch import nn

from tablebench.models.torchutils import evaluate


def append_by_key(from_dict: dict, to_dict: Union[dict, defaultdict]) -> dict:
    for k, v in from_dict.items():
        assert (k in to_dict) or (isinstance(to_dict, defaultdict))
        to_dict[k].append(v)
    return to_dict


class SklearnStylePytorchModel(ABC, nn.Module):
    """A pytorch model with an sklearn-style interface."""

    def predict(self, X) -> np.ndarray:
        """sklearn-compatible prediction function."""
        return self(X).detach().cpu().numpy()

    @abstractmethod
    def predict_proba(self, X) -> np.ndarray:
        """sklearn-compatible probability prediction function."""
        raise

    def evaluate(self, train_loader, other_loaders, device):
        split_scores = {"train": evaluate(self, train_loader, device)}
        if other_loaders:
            for split, loader in other_loaders.items():
                split_score = evaluate(self, loader, device)
                split_scores[split] = split_score
        return split_scores

    @abstractmethod
    def train_epoch(self, train_loader: torch.utils.data.DataLoader,
                    optimizer: torch.optim.Optimizer,
                    loss_fn: Callable,
                    device: str,
                    other_loaders: Optional[
                        Mapping[str, torch.utils.data.DataLoader]] = None
                    ):
        """Conduct one epoch of training."""
        raise

    def save_checkpoint(self, optimizer: torch.optim.Optimizer) -> Checkpoint:
        # Here we save a checkpoint. It is automatically registered with
        # Ray Tune and can be accessed through `session.get_checkpoint()`
        # API in future iterations.
        os.makedirs("model", exist_ok=True)
        torch.save(
            (self.state_dict(), optimizer.state_dict()),
            "model/checkpoint.pt")
        checkpoint = Checkpoint.from_directory("model")
        return checkpoint

    def fit(self,
            train_loader: torch.utils.data.DataLoader,
            optimizer: torch.optim.Optimizer,
            loss_fn,
            device: str,
            n_epochs=1,
            other_loaders: Optional[
                Mapping[str, torch.utils.data.DataLoader]] = None,
            tune_report_split: Optional[str] = None) -> dict:
        fit_metrics = defaultdict(list)

        if tune_report_split:
            assert tune_report_split in list(other_loaders.keys()) + ["train"]

        for epoch in range(1, n_epochs + 1):
            self.train_epoch(train_loader=train_loader,
                             optimizer=optimizer,
                             loss_fn=loss_fn,
                             other_loaders=other_loaders,
                             device=device)
            metrics = self.evaluate(train_loader, other_loaders, device=device)
            log_str = f'Epoch {epoch:03d} ' + ' | '.join(
                f"{k} score: {v:.4f}" for k, v in metrics.items())
            print(log_str)

            checkpoint = self.save_checkpoint(optimizer)

            if tune_report_split:
                # TODO(jpgard): consider reporting multiple named metrics here.
                #  e.g. session.report(metrics) or a small subset of metrics.
                session.report({"metric": metrics[tune_report_split]},
                               checkpoint=checkpoint)

            fit_metrics = append_by_key(from_dict=metrics, to_dict=fit_metrics)

        return fit_metrics


SKLEARN_MODEL_NAMES = ("expgrad", "histgbm", "lightgbm", "wcs", "xgb")
PYTORCH_MODEL_NAMES = ("ft_transformer", "group_dro", "mlp", "resnet")


def is_pytorch_model_name(model: str) -> bool:
    """Helper function to determine whether a model name is a pytorch model.

    ISee description of is_pytorch_model() above."""
    is_sklearn = model in SKLEARN_MODEL_NAMES
    is_pt = model in PYTORCH_MODEL_NAMES
    assert is_sklearn or is_pt, f"unknown model name {model}"
    return is_pt

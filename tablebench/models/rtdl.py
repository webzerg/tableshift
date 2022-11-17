"""
Wrappers for tabular baseline models from the rtdl package.

rtdl source: https://github.com/Yura52/rtdl
"""

from typing import Mapping, Optional
import rtdl
import sklearn
import numpy as np
import scipy
import torch

from tablebench.models.compat import SklearnStylePytorchModel


def apply_model(model: torch.nn.Module, x_num, x_cat=None):
    if isinstance(model, rtdl.FTTransformer):
        return model(x_num, x_cat)
    elif isinstance(model, (rtdl.MLP, rtdl.ResNet)):
        assert x_cat is None
        return model(x_num)
    else:
        raise NotImplementedError(
            f'Looks like you are using a custom model: {type(model)}.'
            ' Then you have to implement this branch first.'
        )


@torch.no_grad()
def evaluate(model, loader):
    model.eval()
    prediction = []
    label = []
    for (batch_x, batch_y, _) in loader:
        # TODO(jpgard): handle categorical features here.
        prediction.append(apply_model(model, batch_x))
        label.append(batch_y.squeeze())
    prediction = torch.cat(prediction).squeeze(1).cpu().numpy()
    target = torch.cat(label).squeeze().cpu().numpy()

    prediction = np.round(scipy.special.expit(prediction))
    score = sklearn.metrics.accuracy_score(target, prediction)
    return score


def train_epoch(model: torch.nn.Module,
                train_loader: torch.utils.data.DataLoader,
                optimizer: torch.optim.Optimizer,
                loss_fn,
                other_loaders: Optional[
                    Mapping[str, torch.utils.data.DataLoader]] = None):
    """Run a single epoch of model training."""
    for iteration, (x_batch, y_batch, _) in enumerate(train_loader):
        model.train()
        optimizer.zero_grad()
        # TODO(jpgard): handle categorical features here.
        loss = loss_fn(apply_model(model, x_batch).squeeze(1), y_batch)
        loss.backward()
        optimizer.step()
    split_scores = {"train": evaluate(model, train_loader)}
    if other_loaders:
        for split, loader in other_loaders.items():
            split_score = evaluate(model, loader)
            split_scores[split] = split_score
    return split_scores


@torch.no_grad()
def predict_proba(model, X):
    prediction = model.predict(X)
    return scipy.special.expit(prediction)


class SklearnStyleRTDLModel(SklearnStylePytorchModel):
    def fit(self,
            train_loader: torch.utils.data.DataLoader,
            optimizer: torch.optim.Optimizer,
            loss_fn,
            n_epochs=1,
            other_loaders: Optional[
                Mapping[str, torch.utils.data.DataLoader]] = None):
        for epoch in range(1, n_epochs + 1):
            metrics = train_epoch(self, train_loader=train_loader,
                                  optimizer=optimizer,
                                  loss_fn=loss_fn, other_loaders=other_loaders)
            log_str = f'Epoch {epoch:03d} ' + ' | '.join(
                f"{k} score: {v:.4f}" for k, v in metrics.items())
            print(log_str)


class ResNetModel(rtdl.ResNet, SklearnStyleRTDLModel):

    def predict_proba(self, X) -> np.ndarray:
        return predict_proba(self, X)


class MLPModel(rtdl.MLP, SklearnStyleRTDLModel):
    def predict_proba(self, X) -> np.ndarray:
        return predict_proba(self, X)


class FTTransformerModel(rtdl.FTTransformer, SklearnStyleRTDLModel):
    def predict_proba(self, X) -> np.ndarray:
        return predict_proba(self, X)

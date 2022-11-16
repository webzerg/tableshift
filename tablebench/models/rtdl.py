from typing import Mapping, Optional
import rtdl
import sklearn
import numpy as np
import scipy
import torch

from tablebench.models.compat import SklearnStylePytorchModel


# def apply_model(x_num, x_cat=None):
#     if isinstance(model, rtdl.FTTransformer):
#         return model(x_num, x_cat)
#     elif isinstance(model, (rtdl.MLP, rtdl.ResNet)):
#         assert x_cat is None
#         return model(x_num)
#     else:
#         raise NotImplementedError(
#             f'Looks like you are using a custom model: {type(model)}.'
#             ' Then you have to implement this branch first.'
#         )


@torch.no_grad()
def evaluate(model, loader):
    model.eval()
    prediction = []
    label = []
    for (batch_x, batch_y, _) in loader:
        prediction.append(model(batch_x))
        label.append(batch_y.squeeze())
    prediction = torch.cat(prediction).squeeze(1).cpu().numpy()
    target = torch.cat(label).squeeze().cpu().numpy()

    prediction = np.round(scipy.special.expit(prediction))
    score = sklearn.metrics.accuracy_score(target, prediction)
    return score


def fit(model: torch.nn.Module,
        train_loader: torch.utils.data.DataLoader,
        optimizer: torch.optim.Optimizer,
        loss_fn,
        n_epochs=1,
        other_loaders: Optional[
            Mapping[str, torch.utils.data.DataLoader]] = None):
    """Fit a model."""
    for epoch in range(1, n_epochs + 1):
        for iteration, (x_batch, y_batch, _) in enumerate(train_loader):
            model.train()
            optimizer.zero_grad()
            loss = loss_fn(model(x_batch).squeeze(1), y_batch)
            loss.backward()
            optimizer.step()
            print(f'(epoch) {epoch} (batch) {iteration} '
                  f'(loss) {loss.item():.4f}')
        split_scores = {}
        log_str = f'Epoch {epoch:03d}'
        if other_loaders:
            for split, loader in other_loaders.items():
                split_score = evaluate(model, loader)
                split_scores[split] = split_score
                log_str += f' | {split} score: {split_score:.4f}'

        print(log_str, end='')


@torch.no_grad()
def predict_proba(model, X):
    prediction = model.predict(X)
    return scipy.special.expit(prediction)


class ResNetModel(rtdl.ResNet, SklearnStylePytorchModel):

    def predict_proba(self, X) -> np.ndarray:
        return predict_proba(self, X)

    def fit(self,
            train_loader: torch.utils.data.DataLoader,
            optimizer: torch.optim.Optimizer,
            loss_fn,
            n_epochs=1,
            other_loaders: Optional[
                Mapping[str, torch.utils.data.DataLoader]] = None):
        fit(self, train_loader=train_loader, optimizer=optimizer,
            loss_fn=loss_fn, n_epochs=n_epochs,
            other_loaders=other_loaders)


class MLPModel(rtdl.MLP, SklearnStylePytorchModel):
    def predict_proba(self, X) -> np.ndarray:
        return predict_proba(self, X)

    def fit(self,
            train_loader: torch.utils.data.DataLoader,
            optimizer: torch.optim.Optimizer,
            loss_fn,
            n_epochs=1,
            other_loaders: Optional[
                Mapping[str, torch.utils.data.DataLoader]] = None):
        fit(self, train_loader=train_loader, optimizer=optimizer,
            loss_fn=loss_fn, n_epochs=n_epochs,
            other_loaders=other_loaders)

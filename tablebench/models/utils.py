from typing import Tuple
import numpy as np
import rtdl
import scipy
import sklearn
import torch


@torch.no_grad()
def get_predictions_and_labels(model, loader, as_logits=False) -> Tuple[
    np.ndarray, np.ndarray]:
    """Get the predictions (as logits, or probabilities) and labels."""
    prediction = []
    label = []

    for (batch_x, batch_y, _) in loader:
        # TODO(jpgard): handle categorical features here.
        prediction.append(apply_model(model, batch_x))
        label.append(batch_y.squeeze())
    prediction = torch.cat(prediction).squeeze(1).cpu().numpy()
    target = torch.cat(label).squeeze().cpu().numpy()
    if not as_logits:
        prediction = scipy.special.expit(prediction)
    return prediction, target


@torch.no_grad()
def evaluate(model, loader):
    model.eval()
    prediction, target = get_predictions_and_labels(model, loader)
    prediction = np.round(prediction)
    score = sklearn.metrics.accuracy_score(target, prediction)
    return score


def apply_model(model: torch.nn.Module, x_num, x_cat=None):
    if isinstance(model, rtdl.FTTransformer):
        return model(x_num, x_cat)
    elif isinstance(model, (rtdl.MLP, rtdl.ResNet)):
        assert x_cat is None
        return model(x_num)
    else:
        raise NotImplementedError(
            f'[ERROR] Looks like you are using a custom model: {type(model)}.'
            ' Then you have to implement this branch first.'
        )

from typing import Any

import rtdl
import torch
from torch.nn import functional as F
from frozendict import frozendict

from tablebench.core import TabularDataset
from tablebench.models import GroupDROModel
from tablebench.models.dro import group_dro_loss
from tablebench.models import is_pytorch_model
from tablebench.models.expgrad import ExponentiatedGradient

PYTORCH_DEFAULTS = frozendict({
    "lr": 0.001,
    "weight_decay": 0.0,
    "n_epochs": 1,
    "batch_size": 512,
})


def _train_pytorch(estimator, dset: TabularDataset, device: str,
                   config=PYTORCH_DEFAULTS):
    """Helper function to train a pytorch estimator."""

    train_loader = dset.get_dataloader("train", config["batch_size"],
                                       device=device)
    eval_loaders = {
        s: dset.get_dataloader(s, config["batch_size"], device=device) for s in
        dset.eval_split_names}

    loss_fn = (group_dro_loss
               if isinstance(estimator, GroupDROModel)
               else F.binary_cross_entropy_with_logits)

    optimizer = (
        estimator.make_default_optimizer()
        if isinstance(estimator, rtdl.FTTransformer)
        else torch.optim.AdamW(estimator.parameters(), lr=config["lr"],
                               weight_decay=config["weight_decay"])
    )
    estimator.to(device)
    estimator.fit(train_loader, optimizer, loss_fn, n_epochs=config["n_epochs"],
                  other_loaders=eval_loaders)
    return estimator


def _train_sklearn(estimator, dset: TabularDataset):
    """Helper function to train a sklearn-type estimator."""
    print(f"fitting estimator of type {type(estimator)}")
    X_tr, y_tr, _, d_tr = dset.get_pandas(split="train")
    if isinstance(estimator, ExponentiatedGradient):
        estimator.fit(X_tr, y_tr, d=d_tr)
    else:
        estimator.fit(X_tr, y_tr)
    print("fitting estimator complete.")

    for split in dset.eval_split_names:

        X_te, _, _, _ = dset.get_pandas(split=split)

        y_hat_te = estimator.predict(X_te)
        metrics = dset.evaluate_predictions(y_hat_te, split=split)
        print(f"metrics on split {split}:")
        for k, v in metrics.items():
            print(f"\t{k:<40}:{v:.3f}")
    return estimator


def train(estimator: Any, dset: TabularDataset, **kwargs):
    if is_pytorch_model(estimator):
        return _train_pytorch(estimator, dset, **kwargs)
    else:
        return _train_sklearn(estimator, dset)

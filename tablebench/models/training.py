import os
from typing import Any

from frozendict import frozendict
from ray.air import session
import rtdl
import torch
from torch.nn import functional as F

from tablebench.core import TabularDataset
from tablebench.models import GroupDROModel
from tablebench.models.compat import SklearnStylePytorchModel
from tablebench.models.dro import group_dro_loss
from tablebench.models import is_pytorch_model
from tablebench.models.expgrad import ExponentiatedGradient

PYTORCH_DEFAULTS = frozendict({
    "lr": 0.001,
    "weight_decay": 0.0,
    "n_epochs": 1,
    "batch_size": 512,
})


def get_optimizer(estimator: SklearnStylePytorchModel,
                  config) -> torch.optim.Optimizer:
    optimizer = (
        estimator.make_default_optimizer()
        if isinstance(estimator, rtdl.FTTransformer)
        else torch.optim.AdamW(estimator.parameters(), lr=config["lr"],
                               weight_decay=config["weight_decay"])
    )
    return optimizer


def _train_pytorch(estimator: SklearnStylePytorchModel, dset: TabularDataset,
                   device: str,
                   config=PYTORCH_DEFAULTS):
    """Helper function to train a pytorch estimator."""
    optimizer = get_optimizer(estimator, config)

    print(f"[DEBUG] config is {config}")
    print(f"[DEBUG] device is {device}")

    train_loader = dset.get_dataloader("train", config["batch_size"],
                                       device=device)
    eval_loaders = {
        s: dset.get_dataloader(s, config["batch_size"], device=device) for s in
        dset.eval_split_names}

    loss_fn = (group_dro_loss
               if isinstance(estimator, GroupDROModel)
               else F.binary_cross_entropy_with_logits)

    # To restore a checkpoint, use `session.get_checkpoint()`.
    loaded_checkpoint = session.get_checkpoint()
    if loaded_checkpoint:
        with loaded_checkpoint.as_directory() as loaded_checkpoint_dir:
            model_state, optimizer_state = torch.load(
                os.path.join(loaded_checkpoint_dir, "checkpoint.pt"))
            estimator.load_state_dict(model_state)
            optimizer.load_state_dict(optimizer_state)

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

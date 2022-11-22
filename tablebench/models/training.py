import rtdl
import torch
from torch.nn import functional as F

from tablebench.models import GroupDROModel
from tablebench.models.dro import group_dro_loss

PYTORCH_DEFAULTS = {
    "lr": 0.001,
    "weight_decay": 0.0,
    "n_epochs": 1,
    "batch_size": 512,
}


def train_pytorch(estimator, dset, device, config=PYTORCH_DEFAULTS):
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


def train_sklearn(estimator, dset):
    """Helper function to train a sklearn-type estimator."""
    print(f"fitting estimator of type {type(estimator)}")
    X_tr, y_tr, G_tr = dset.get_pandas(split="train")
    estimator.fit(X_tr, y_tr)
    print("fitting estimator complete.")

    for split in dset.eval_split_names:

        X_te, _, _ = dset.get_pandas(split=split)

        y_hat_te = estimator.predict(X_te)
        metrics = dset.evaluate_predictions(y_hat_te, split=split)
        print(f"metrics on split {split}:")
        for k, v in metrics.items():
            print(f"\t{k:<40}:{v:.3f}")
    return estimator

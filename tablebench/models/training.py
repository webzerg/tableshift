import rtdl
import torch
from torch.nn import functional as F

from tablebench.models import GroupDROModel
from tablebench.models.dro import group_dro_loss


def train_pytorch(model, dset, device, eval_splits=("validation", "test")):
    """Helper function to train a pytorch model."""
    train_loader = dset.get_dataloader("train", 512, device=device)
    eval_loaders = {s: dset.get_dataloader(s, 2048, device=device) for s in
                    eval_splits}

    config = {
        "lr": 0.001,
        "weight_decay": 0.0,
    }

    loss_fn = (group_dro_loss
               if isinstance(model, GroupDROModel)
               else F.binary_cross_entropy_with_logits)

    optimizer = (
        model.make_default_optimizer()
        if isinstance(model, rtdl.FTTransformer)
        else torch.optim.AdamW(model.parameters(), lr=config["lr"],
                               weight_decay=config["weight_decay"])
    )
    model.to(device)
    model.fit(train_loader, optimizer, loss_fn, n_epochs=25,
              other_loaders=eval_loaders)
    return


def train_sklearn(estimator, dset, eval_splits=("test",)):
    """Helper function to train a sklearn-type model."""
    print(f"fitting estimator of type {type(estimator)}")
    X_tr, y_tr, G_tr = dset.get_pandas(split="train")
    estimator.fit(X_tr, y_tr)
    print("fitting estimator complete.")

    for split in eval_splits:

        X_te, _, _ = dset.get_pandas(split=split)

        y_hat_te = estimator.predict(X_te)
        metrics = dset.evaluate_predictions(y_hat_te, split=split)
        print(f"metrics on split {split}:")
        for k, v in metrics.items():
            print(f"\t{k:<40}:{v:.3f}")
    return

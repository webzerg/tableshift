import os
from typing import Any

from frozendict import frozendict
from ray.air import session
import torch

from tablebench.core import TabularDataset
from tablebench.models.compat import SklearnStylePytorchModel
from tablebench.models.expgrad import ExponentiatedGradient
from tablebench.models.wcs import WeightedCovariateShiftClassifier
from tablebench.models.torchutils import unpack_batch, apply_model
from tablebench.models.losses import DomainLoss, GroupDROLoss
from tablebench.models.torchutils import get_module_attr

PYTORCH_DEFAULTS = frozendict({
    "lr": 0.001,
    "weight_decay": 0.0,
    "n_epochs": 1,
    "batch_size": 512,
})


def train_epoch(model, optimizer, criterion, train_loader,
                device) -> float:
    """Run one epoch of training, and return the training loss."""

    model.train()
    running_loss = 0.0
    n_train = 0
    for i, batch in enumerate(train_loader):
        # get the inputs and labels
        inputs, labels, _, domains = unpack_batch(batch)
        inputs = inputs.float().to(device)
        labels = labels.float().to(device)

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = apply_model(model, inputs).squeeze()
        if isinstance(criterion, GroupDROLoss):
            # Case: loss requires domain labels, plus group weights + step size.
            domains = domains.float().to(device)
            group_weights = get_module_attr(model, "group_weights").to(device)
            group_weights_step_size = get_module_attr(
                model, "group_weights_step_size").to(device)
            loss = criterion(
                outputs, labels, domains,
                group_weights=group_weights,
                group_weights_step_size=group_weights_step_size,
                device=device)

        elif isinstance(criterion, DomainLoss):
            # Case: loss requires domain labels.
            domains = domains.float()
            loss = criterion(outputs, labels, domains)

        else:
            # Case: standard loss; only requires targets and predictions.
            loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # print statistics
        running_loss += loss.item()
        n_train += len(inputs)

    return running_loss / n_train


def _train_pytorch(estimator: SklearnStylePytorchModel, dset: TabularDataset,
                   device: str,
                   config=PYTORCH_DEFAULTS,
                   tune_report_split: str = None):
    """Helper function to train a pytorch estimator."""
    print(f"[DEBUG] config is {config}")
    print(f"[DEBUG] device is {device}")
    print(f"[DEBUG] tune_report_split is {tune_report_split}")

    # TODO(jpgard): clean up this block; should be an if/else
    train_loader = dset.get_dataloader("train", config["batch_size"])
    domain_loaders = dset.get_domain_dataloaders("train", config["batch_size"])

    eval_loaders = {s: dset.get_dataloader(s, config["batch_size"]) for s in
                    dset.eval_split_names}
    train_eval_loaders = dset.get_domain_dataloaders("train",
                                                     config["batch_size"],
                                                     infinite=False)
    eval_loaders.update(train_eval_loaders)

    loss_fn = config["criterion"]

    estimator.to(device)
    # estimator.fit(train_loader, loss_fn,
    #               n_epochs=config["n_epochs"],
    #               device=device,
    #               eval_loaders=eval_loaders,
    #               tune_report_split=tune_report_split)
    estimator.fit(domain_loaders, loss_fn,
                  n_epochs=config["n_epochs"],
                  device=device,
                  eval_loaders=eval_loaders,
                  tune_report_split=tune_report_split,
                  max_examples_per_epoch=dset.n_train)
    return estimator


def _train_sklearn(estimator, dset: TabularDataset,
                   tune_report_split: str = None):
    """Helper function to train a sklearn-type estimator."""
    X_tr, y_tr, _, d_tr = dset.get_pandas(split="train")
    if isinstance(estimator, ExponentiatedGradient):
        estimator.fit(X_tr, y_tr, d=d_tr)
    elif isinstance(estimator, WeightedCovariateShiftClassifier):
        X_ood_tr, y_ood_tr, _, _ = dset.get_pandas(split="ood_validation")
        estimator.fit(X_tr, y_tr, X_ood_tr)
    else:
        estimator.fit(X_tr, y_tr)
    print("fitting estimator complete.")

    if tune_report_split:
        X_te, _, _, _ = dset.get_pandas(split=tune_report_split)
        y_hat_te = estimator.predict(X_te)
        metrics = dset.evaluate_predictions(y_hat_te, split=tune_report_split)
        session.report({"metric": metrics[f"accuracy_{tune_report_split}"]})
    return estimator


def train(estimator: Any, dset: TabularDataset, tune_report_split: str = None,
          **kwargs):
    print(f"fitting estimator of type {type(estimator)}")
    if isinstance(estimator, torch.nn.Module):
        assert isinstance(
            estimator,
            SklearnStylePytorchModel), \
            f"train() can only be called with SklearnStylePytorchModel; got " \
            f"type {type(estimator)} "
        return _train_pytorch(estimator, dset,
                              tune_report_split=tune_report_split, **kwargs)
    else:
        return _train_sklearn(estimator, dset,
                              tune_report_split=tune_report_split)

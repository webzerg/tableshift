from dataclasses import dataclass
from functools import partial
from typing import Dict, Any, List

import fairlearn.reductions
import numpy as np
import pandas as pd
import ray
import sklearn
import torch
from ray import train, tune
from ray.air import session, ScalingConfig, RunConfig
from ray.train.lightgbm import LightGBMTrainer
from ray.train.torch import TorchCheckpoint, TorchTrainer
from ray.train.xgboost import XGBoostTrainer
from ray.tune import Tuner
from ray.tune.schedulers import ASHAScheduler
from ray.tune.search.hyperopt import HyperOptSearch

from tablebench.configs.hparams import search_space
from tablebench.core import TabularDataset
from tablebench.models.compat import SklearnStylePytorchModel, \
    is_pytorch_model_name
from tablebench.models.config import get_default_config
from tablebench.models.expgrad import ExponentiatedGradientTrainer
from tablebench.models.torchutils import get_predictions_and_labels
from tablebench.models.training import get_optimizer, train_epoch
from tablebench.models.utils import get_estimator


@dataclass
class TuneConfig:
    """Container for various Ray tuning parameters.

    Note that this is different from the Ray TuneConfig class, as it actually
    contains parameters that are passed to different parts of the ray API
    such as `ScalingConfig`, which consumes the num_workers."""
    max_concurrent_trials: int
    num_workers: int = 1
    num_samples: int = 1
    tune_metric_name: str = "metric"
    tune_metric_higher_is_better: bool = True
    early_stop: bool = True

    @property
    def mode(self):
        return "max" if self.tune_metric_higher_is_better else "min"


def make_ray_dataset(dset: TabularDataset, split, keep_domain_labels=False):
    X, y, G, d = dset.get_pandas(split)
    if (d is None) or (not keep_domain_labels):
        df = pd.concat([X, y, G], axis=1)
    else:
        df = pd.concat([X, y, G, d], axis=1)
    df = df.loc[:, ~df.columns.duplicated()].copy()

    dataset: ray.data.Dataset = ray.data.from_pandas([df])
    return dataset


def get_ray_checkpoint(model):
    if isinstance(model, torch.nn.parallel.DistributedDataParallel):
        return TorchCheckpoint.from_state_dict(model.module.state_dict())
    else:
        return TorchCheckpoint.from_state_dict(model.state_dict())


def ray_evaluate(model, splits: Dict[str, Any]) -> dict:
    """Run evaluation of a model.

    splits should be a dict mapping split names to DataLoaders.
    """
    device = train.torch.get_device()
    model.eval()
    metrics = {}
    for split in splits:
        prediction, target = get_predictions_and_labels(model, splits[split],
                                                        device=device)
        prediction = np.round(prediction)
        acc = sklearn.metrics.accuracy_score(target, prediction)
        metrics[f"{split}_accuracy"] = acc
    return metrics


def _row_to_dict(row, X_names: List[str], y_name: str, G_names: List[str],
                 d_name: str) -> Dict:
    """Convert ray PandasRow to a dict of numpy arrays."""
    x = row[X_names].values.astype(float)
    y = row[y_name].values.astype(float)
    g = row[G_names].values.astype(float)
    outputs = {"x": x, "y": y, "g": g}
    if d_name in row:
        outputs["d"] = row[d_name].values.astype(float)
    return outputs


def prepare_torch_datasets(split, dset: TabularDataset):
    keep_domain_labels = dset.domain_label_colname is not None
    ds = make_ray_dataset(dset, split, keep_domain_labels)

    y_name = dset.target
    d_name = dset.domain_label_colname
    G_names = dset.group_feature_names
    X_names = dset.feature_names

    _map_fn = partial(_row_to_dict, X_names=X_names, y_name=y_name,
                      G_names=G_names, d_name=d_name)

    return ds.map_batches(_map_fn, batch_format="pandas")


def run_ray_tune_experiment(dset: TabularDataset,
                            model_name: str,
                            tune_config: TuneConfig = None,
                            max_epochs=100):
    """Rune a ray tuning experiment.

    This defines the trainers, tuner, and other associated objects, runs the
    tuning experiment, and returns the ray ResultGrid object.
    """
    def train_loop_per_worker(config: Dict):
        """Function to be run by each TorchTrainer.

        Must be defined inside main() because this function can only have a
        single argument, named config, but it also requires the use of the
        model_name command-line flag.
        """
        model = get_estimator(model_name, **config)
        assert isinstance(model, SklearnStylePytorchModel)
        model = train.torch.prepare_model(model)

        criterion = config["criterion"]
        optimizer = get_optimizer(model, config)

        n_epochs = config["n_epochs"] \
            if not tune_config.early_stop else max_epochs

        device = train.torch.get_device()

        for epoch in range(n_epochs):
            print(f"[DEBUG] starting epoch {epoch}")

            train_dataset_batches = session.get_dataset_shard(
                "train").iter_torch_batches(batch_size=config["batch_size"])
            eval_batches = {
                split: session.get_dataset_shard(split).iter_torch_batches(
                    batch_size=config["batch_size"]) for split in dset.splits}

            train_loss = train_epoch(model, optimizer, criterion,
                                     train_dataset_batches, device=device)
            metrics = ray_evaluate(model, eval_batches)

            # Log the metrics for this epoch
            metrics.update(dict(train_loss=train_loss))
            checkpoint = get_ray_checkpoint(model)
            session.report(metrics, checkpoint=checkpoint)

    # Get the default/fixed configs (these are provided to every Trainer but
    # can be overwritten if they are also in the param_space).
    default_train_config = get_default_config(model_name, dset)
    scaling_config = ScalingConfig(
        num_workers=tune_config.num_workers,
        use_gpu=torch.cuda.is_available())

    # Construct the Trainer object that will be passed to each worker.
    if is_pytorch_model_name(model_name):
        datasets = {split: prepare_torch_datasets(split, dset) for split in
                    dset.splits}

        trainer = TorchTrainer(
            train_loop_per_worker=train_loop_per_worker,
            train_loop_config=default_train_config,
            datasets=datasets,
            scaling_config=scaling_config)
        # Hyperparameter search space; note that the scaling_config can also
        # be tuned but is fixed here.
        param_space = {
            # The params will be merged with the ones defined in the Trainer.
            "train_loop_config": search_space[model_name],
            # Optionally, could tune the number of distributed workers here.
            # "scaling_config": ScalingConfig(num_workers=2)
        }

    elif model_name == "xgb":
        datasets = {split: make_ray_dataset(dset, split) for split in
                    dset.splits}
        params = {
            "tree_method": "gpu_hist" if torch.cuda.is_available() else "hist",
            "objective": "binary:logistic",
            "eval_metric": "error"}
        trainer = XGBoostTrainer(label_column=dset.target,
                                 datasets=datasets,
                                 params=params,
                                 scaling_config=scaling_config)
        tune_config.tune_metric_name = "validation-error"
        tune_config.tune_metric_higher_is_better = False
        param_space = {"params": search_space[model_name]}

    elif model_name == "lightgbm":

        datasets = {split: make_ray_dataset(dset, split) for split in
                    dset.splits}
        params = {"objective": "binary",
                  "metric": "binary_error",
                  "device_type": "gpu" if torch.cuda.is_available() else "cpu"}
        trainer = LightGBMTrainer(label_column=dset.target,
                                  datasets=datasets,
                                  params=params,
                                  scaling_config=scaling_config)
        param_space = {"params": search_space[model_name]}
        tune_config.tune_metric_name = "validation-binary_error"
        tune_config.tune_metric_higher_is_better = False

    elif model_name == "expgrad":
        # This currently does not run; there isn't a way to scale this
        # to scale due to need for entire dataset in-memory in
        # ExponentiatedGradient.
        import fairlearn.reductions
        # This trainer should only run for datasets with domain splits.
        assert dset.domain_label_colname
        datasets = {split: make_ray_dataset(dset, split, True) for split in
                    dset.splits}
        trainer = ExponentiatedGradientTrainer(
            label_column=dset.target,
            domain_column=dset.domain_label_colname,
            feature_columns=dset.feature_names,
            datasets=datasets,
            params={"constraints": fairlearn.reductions.ErrorRateParity()},
        )

    else:
        raise NotImplementedError(f"model {model_name} not implemented.")

    if tune_config is None:
        print("[DEBUG] no TuneConfig provided; no tuning will be performed.")
        # To run just a single training iteration (without tuning)
        result = trainer.fit()
        latest_checkpoint = result.checkpoint
        return result

    # Create Tuner.

    stopper = tune.stopper.ExperimentPlateauStopper(
        metric=tune_config.tune_metric_name,
        mode=tune_config.mode,
        # TODO(jpgard): increase patience to 16 as in
        #  https://arxiv.org/pdf/2106.11959.pdf
        patience=5) if tune_config.early_stop else None

    tuner = Tuner(
        trainable=trainer,
        run_config=RunConfig(name="test_tuner_notebook",
                             local_dir="ray-results",
                             stop=stopper),
        param_space=param_space,
        tune_config=tune.TuneConfig(
            search_alg=HyperOptSearch(metric=tune_config.tune_metric_name,
                                      mode=tune_config.mode),
            scheduler=ASHAScheduler(
                time_attr='training_iteration',
                metric=tune_config.tune_metric_name,
                mode=tune_config.mode,
                stop_last_trials=True),
            num_samples=tune_config.num_samples,
            max_concurrent_trials=tune_config.max_concurrent_trials))

    results = tuner.fit()

    return results

from dataclasses import dataclass
from functools import partial
import re
from typing import Dict, Any, List, Union, Tuple

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
from tablebench.core import TabularDataset, CachedDataset
from tablebench.models.compat import SklearnStylePytorchModel, \
    is_pytorch_model_name
from tablebench.models.config import get_default_config
from tablebench.models.expgrad import ExponentiatedGradientTrainer
from tablebench.models.torchutils import get_predictions_and_labels
from tablebench.models.training import get_optimizer, train_epoch
from tablebench.models.utils import get_estimator


def accuracy_metric_name_and_mode_for_model(model_name: str, split="validation") -> Tuple[str, str]:
    """Helper function to fetch the name for an accuracy-related metric for each model.

    This is necessary because some Ray Trainer types do not allow for custom naming of the metrics, and
    so we may need to minimize error <-> maximize accuracy depending on the trainer type.
    """
    if model_name == "xgb":
        metric_name = f"{split}-error"
        mode = "min"
    elif model_name == "lightgbm":
        metric_name = f"{split}-binary_error"
        mode = "min"
    elif is_pytorch_model_name(model_name):
        metric_name = f"{split}_accuracy"
        mode = "max"
    else:
        raise NotImplementedError(f"cannot find accuracy metric name for model {model_name}")
    return metric_name, mode


_DEFAULT_RANDOM_STATE = 449237829


@dataclass
class RayExperimentConfig:
    """Container for various Ray tuning parameters.

    Note that this is different from the Ray TuneConfig class, as it actually
    contains parameters that are passed to different parts of the ray API
    such as `ScalingConfig`, which consumes the num_workers."""
    max_concurrent_trials: int
    mode: str
    num_workers: int = 1
    num_samples: int = 1
    tune_metric_name: str = "metric"
    time_budget_hrs: float = None
    search_alg: str = "hyperopt"
    scheduler: str = None
    random_state: int = _DEFAULT_RANDOM_STATE  # random state for determinism in the search algorithm
    gpu_per_worker: float = 1.0  # set to fraction to allow multiple workers per GPU

    def get_search_alg(self):
        print(f"[INFO] instantiating search alg of type {self.search_alg}")
        if self.search_alg == "hyperopt":
            return HyperOptSearch(metric=self.tune_metric_name,
                                  mode=self.mode, random_state_seed=self.random_state)
        elif self.search_alg == "random":
            return tune.search.basic_variant.BasicVariantGenerator(max_concurrent=self.max_concurrent_trials,
                                                                   random_state=self.random_state)
        else:
            raise NotImplementedError

    def get_scheduler(self):
        if self.scheduler is None:
            return None
        elif self.scheduler == "asha":
            return ASHAScheduler(
                time_attr='training_iteration',
                metric=self.tune_metric_name,
                mode=self.mode,
                stop_last_trials=True)
        elif self.scheduler == "median":
            return tune.schedulers.MedianStoppingRule(
                time_attr='training_iteration',
                metric=self.tune_metric_name,
                mode=self.mode,
                grace_period=5  # default is 60 (seconds); needs to be set.
            )
        else:
            raise NotImplementedError


def make_ray_dataset(dset: Union[TabularDataset, CachedDataset], split, keep_domain_labels=False):
    if isinstance(dset, CachedDataset):
        return dset.get_ray(split)
    else:
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
        metrics[f"{split}_ymean"] = np.mean(target).item()
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


def prepare_torch_datasets(split, dset: Union[TabularDataset, CachedDataset]):
    keep_domain_labels = dset.domain_label_colname is not None

    if isinstance(dset, TabularDataset):
        ds = make_ray_dataset(dset, split, keep_domain_labels)
    elif isinstance(dset, CachedDataset):
        ds = dset.get_ray(split)
    y_name = dset.target
    d_name = dset.domain_label_colname
    G_names = dset.group_feature_names
    X_names = dset.feature_names

    _map_fn = partial(_row_to_dict, X_names=X_names, y_name=y_name,
                      G_names=G_names, d_name=d_name)

    return ds.map_batches(_map_fn, batch_format="pandas")


def run_ray_tune_experiment(dset: Union[TabularDataset, CachedDataset],
                            model_name: str,
                            tune_config: RayExperimentConfig = None,
                            max_epochs=100, debug=False):
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

        if debug:
            # In debug mode,  train only for 2 epochs (2, not 1, so that we can ensure DataLoaders are
            # iterating properly).
            n_epochs = 2

        device = train.torch.get_device()

        for epoch in range(n_epochs):
            print(f"[DEBUG] starting epoch {epoch} with model {model_name}")

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

    # Construct the Trainer object that will be passed to each worker.
    if is_pytorch_model_name(model_name):

        datasets = {split: prepare_torch_datasets(split, dset) for split in dset.splits}

        use_gpu = torch.cuda.is_available()
        trainer = TorchTrainer(
            train_loop_per_worker=train_loop_per_worker,
            train_loop_config=default_train_config,
            datasets=datasets,
            scaling_config=ScalingConfig(
                num_workers=tune_config.num_workers,
                resources_per_worker={"GPU": tune_config.gpu_per_worker} if use_gpu else None,
                use_gpu=use_gpu))
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
        scaling_config = ScalingConfig(
            num_workers=tune_config.num_workers,
            use_gpu=False)
        params = {
            # Note: tree_method must be gpu_hist if using GPU.
            "tree_method": "hist",
            "objective": "binary:logistic",
            "eval_metric": "error"}
        trainer = XGBoostTrainer(label_column=dset.target,
                                 datasets=datasets,
                                 params=params,
                                 scaling_config=scaling_config)
        param_space = {"params": search_space[model_name]}

    elif model_name == "lightgbm":
        print("[WARNING] overriding scaling config for LightGBM; GPU not "
              "currently supported.")
        scaling_config = ScalingConfig(
            num_workers=tune_config.num_workers,
            use_gpu=False)
        datasets = {split: make_ray_dataset(dset, split) for split in
                    dset.splits}
        params = {"objective": "binary",
                  "metric": "binary_error",
                  # Note: device_type must be 'gpu' if using GPU.
                  "device_type": "cpu"}
        trainer = LightGBMTrainer(label_column=dset.target,
                                  datasets=datasets,
                                  params=params,
                                  scaling_config=scaling_config)
        param_space = {"params": search_space[model_name]}

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

    tuner = Tuner(
        trainable=trainer,
        run_config=RunConfig(name="tableshift",
                             local_dir="ray-results"),
        param_space=param_space,
        tune_config=tune.TuneConfig(
            search_alg=tune_config.get_search_alg(),
            scheduler=tune_config.get_scheduler(),
            num_samples=tune_config.num_samples,
            time_budget_s=tune_config.time_budget_hrs * 3600 if tune_config.time_budget_hrs else None,
            max_concurrent_trials=tune_config.max_concurrent_trials))

    results = tuner.fit()

    return results


def fetch_postprocessed_results_df(results: ray.tune.ResultGrid) -> pd.DataFrame:
    """Fetch a DataFrame and clean up the names of columns so they align across models.

    This function accounts for the fact that some Trainers in ray produce results that have the right
    metrics, but with names that don't align to our custom trainers, or they provide error when accuracy
    is desired.
    """
    df = results.get_dataframe()
    for c in df.columns:
        if "error" in c:
            # Replace 'error' columns with 'accuracy' columns.
            # LightGBM uses "{SPLIT}-binary-error"; xgb uses "{SPLIT}-error"
            new_colname = re.sub("-\\w*_*error$", "_accuracy", c)
            df[new_colname] = 1. - df[c]
            df.drop(columns=[c], inplace=True)
    return df

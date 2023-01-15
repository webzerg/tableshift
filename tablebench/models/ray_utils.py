from dataclasses import dataclass
from functools import partial
import gc
import os
import psutil
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
from tqdm import tqdm

from tablebench.configs.hparams import search_space
from tablebench.core import TabularDataset, CachedDataset
from tablebench.models.compat import is_pytorch_model_name, \
    is_domain_generalization_model_name
from tablebench.models.config import get_default_config
from tablebench.models.expgrad import ExponentiatedGradientTrainer
from tablebench.models.torchutils import get_predictions_and_labels, \
    get_module_attr
from tablebench.models.utils import get_estimator


def auto_garbage_collect(pct=75.0, force=False):
    """
    Call the garbage collection if memory used is greater than 80% of total
    available memory. This is called to deal with an issue in Ray not freeing
    up used memory. See https://stackoverflow.com/a/60240396/5843188

    pct - Default value of 80%.  Amount of memory in use that triggers the
    garbage collection call.
    """
    memory_pct = psutil.virtual_memory().percent
    if (memory_pct >= pct) or force:
        print(
            f"[INFO] running garbage collection; memory used "
            f"{psutil.virtual_memory().percent}% "
            f"; threshold is {pct}%; force is {force}.")
        gc.collect()
    else:
        print(f"[INFO] not running garbage collection; "
              f"memory used {psutil.virtual_memory().percent}% < {pct} "
              f"threshold.")
    return


def accuracy_metric_name_and_mode_for_model(model_name: str,
                                            split="validation") -> Tuple[
    str, str]:
    """Fetch the name for an accuracy-related metric for each model.

    This is necessary because some Ray Trainer types do not allow for custom
    naming of the metrics, and so we may need to minimize error <-> maximize
    accuracy depending on the trainer type.
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
        raise NotImplementedError(
            f"cannot find accuracy metric name for model {model_name}")
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
    ray_tmp_dir: str = None
    ray_local_dir: str = None
    num_workers: int = 1
    num_samples: int = 1
    tune_metric_name: str = "metric"
    time_budget_hrs: float = None
    search_alg: str = "hyperopt"
    scheduler: str = None
    random_state: int = _DEFAULT_RANDOM_STATE  # random state for determinism in the search algorithm
    gpu_per_worker: float = 1.0  # set to fraction to allow multiple workers per GPU
    cpu_per_worker: int = 1

    def get_search_alg(self):
        print(f"[INFO] instantiating search alg of type {self.search_alg}")
        if self.search_alg == "hyperopt":
            return HyperOptSearch(metric=self.tune_metric_name,
                                  mode=self.mode,
                                  random_state_seed=self.random_state)
        elif self.search_alg == "random":
            return tune.search.basic_variant.BasicVariantGenerator(
                max_concurrent=self.max_concurrent_trials,
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


def make_ray_dataset(dset: Union[TabularDataset, CachedDataset], split,
                     keep_domain_labels=False, domain=None):
    if isinstance(dset, CachedDataset):
        return dset.get_ray(split, domain=domain)
    else:

        if domain: raise NotImplementedError  # Not currently implemented.

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


def ray_evaluate(model, split_loaders: Dict[str, Any]) -> dict:
    """Run evaluation of a model.

    split_loaders should be a dict mapping split names to DataLoaders.
    """
    dev = train.torch.get_device()
    model.eval()
    metrics = {}
    for split, loader in split_loaders.items():
        prediction_soft, target = get_predictions_and_labels(model, loader, dev)
        prediction_hard = np.round(prediction_soft)
        acc = sklearn.metrics.accuracy_score(target, prediction_hard)
        auc_roc = sklearn.metrics.roc_auc_score(target, prediction_soft)
        avg_prec = sklearn.metrics.average_precision_score(target,
                                                           prediction_soft)
        metrics[f"{split}_accuracy"] = acc
        metrics[f"{split}_auc"] = auc_roc
        metrics[f"{split}_map"] = avg_prec
        metrics[f"{split}_num_samples"] = len(target)
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


def prepare_dataset(split, dset: Union[TabularDataset, CachedDataset],
                    prepare_pytorch: bool,
                    domain: str = None) -> ray.data.Dataset:
    """Prepare a Ray dataset for a specific split (and optional domain)."""
    keep_domain_labels = dset.domain_label_colname is not None

    if isinstance(dset, TabularDataset):
        ds = make_ray_dataset(dset, split, keep_domain_labels)
    elif isinstance(dset, CachedDataset):
        ds = dset.get_ray(split, domain=domain)

    if not prepare_pytorch:
        # Do not need to map batches etc. for non-pytorch datasets.
        return ds

    y_name = dset.target
    d_name = dset.domain_label_colname
    G_names = dset.group_feature_names
    X_names = dset.feature_names

    _map_fn = partial(_row_to_dict, X_names=X_names, y_name=y_name,
                      G_names=G_names, d_name=d_name)

    return ds.map_batches(_map_fn, batch_format="pandas")


def get_per_domain_ray_dsets(dset, split, prepare_pytorch: bool
                             ) -> Dict[str, ray.data.Dataset]:
    dsets = {f"{split}_{domain}": prepare_dataset(
        split, dset, domain=domain, prepare_pytorch=prepare_pytorch)
        for domain in dset.get_domains(split)}
    return dsets


def prepare_ray_datasets(dset: Union[TabularDataset, CachedDataset],
                         split_train_loaders_by_domain: bool,
                         prepare_pytorch: bool,
                         ) -> Dict[str, ray.data.Dataset]:
    """Fetch a dict of {split:ray.data.Dataset} for each split."""
    ray_dsets = {}

    for split in dset.splits:
        if (split == "train" and split_train_loaders_by_domain) \
                or (split in ["id_test", "ood_test"]):
            # Case: prepare per-split dataloaders when training dataset needs
            # to be split by domain (e.g. for domain generalization tasks),
            # and also for the test split of any domain-split task.
            ray_dsets.update(get_per_domain_ray_dsets(dset, split,
                                                      prepare_pytorch))

        ray_dsets[split] = prepare_dataset(split, dset, prepare_pytorch)

    return ray_dsets


def run_ray_tune_experiment(dset: Union[TabularDataset, CachedDataset],
                            model_name: str,
                            tune_config: RayExperimentConfig = None,
                            debug=False):
    """Rune a ray tuning experiment.

    This defines the trainers, tuner, and other associated objects, runs the
    tuning experiment, and returns the ray ResultGrid object.
    """
    auto_garbage_collect()
    dset_domains = {s: dset.get_domains(s) for s in
                    ("train", "test", "id_test", "ood_test")}

    # Explicitly initialize ray in order to set the temp dir.
    ray.init(_temp_dir=tune_config.ray_tmp_dir, ignore_reinit_error=True)

    def train_loop_per_worker(config: Dict):
        """Function to be run by each TorchTrainer.

        Must be defined inside main() because this function can only have a
        single argument, named config, but it also requires the use of the
        model_name command-line flag.
        """
        auto_garbage_collect()
        model = get_estimator(model_name, **config)
        model = train.torch.prepare_model(model)

        criterion = config["criterion"]

        n_epochs = config["n_epochs"]

        if debug:
            # In debug mode,  train only for 2 epochs (2, not 1, so that we
            # can ensure DataLoaders are iterating properly).
            n_epochs = 2

        device = train.torch.get_device()

        def _prepare_dataset_shard(shardname, infinite=False):
            """Get the dataset shard and, optionally, repeat infinitely."""
            shard = session.get_dataset_shard(shardname)
            if infinite:
                print(f"[DEBUG] repeating shard {shardname} infinitely.")
                shard = shard.repeat()
            return shard.iter_torch_batches(batch_size=config["batch_size"])

        for epoch in range(n_epochs):
            print(f"[DEBUG] starting epoch {epoch} with model {model_name}")

            if get_module_attr(model, "domain_generalization"):
                train_loaders = {s: _prepare_dataset_shard(f"train_{s}", True)
                                 for s in dset_domains["train"]}
                uda_loader = None
                max_examples_per_epoch = dset.n_train

            elif get_module_attr(model, "domain_adaptation"):
                raise NotImplementedError
            else:
                train_loaders = {"train": _prepare_dataset_shard("train")}
                uda_loader = None
                max_examples_per_epoch = None

            print(f"[DEBUG] max_examples_per_epoch is {max_examples_per_epoch}")
            print(f"[DEBUG] batch_size is {config['batch_size']}")

            if dset.is_domain_split:
                # Overall eval loaders (compute e.g. overall id/ood test accuracy)
                eval_loaders = {s: _prepare_dataset_shard(s) for s in
                                ('validation', 'id_test', 'ood_test',
                                 'ood_validation')}

                # Per-domain test loaders (for computational efficiency we do not
                # compute per-domain validation metrics).
                id_test_loaders = {s: _prepare_dataset_shard(f"id_test_{s}")
                                   for s in dset_domains['id_test']}
                oo_test_loaders = {s: _prepare_dataset_shard(f"ood_test_{s}")
                                   for s in dset_domains['ood_test']}

                eval_loaders.update(id_test_loaders)
                eval_loaders.update(oo_test_loaders)
            else:
                eval_loaders = {s: _prepare_dataset_shard(s) for s in
                                ('validation', 'test')}

            if isinstance(model, torch.nn.parallel.DistributedDataParallel):
                train_loss = model.module.train_epoch(
                    train_loaders, criterion,
                    device=device, uda_loader=uda_loader,
                    max_examples_per_epoch=max_examples_per_epoch)
            else:
                train_loss = model.train_epoch(
                    train_loaders, criterion,
                    device=device, uda_loader=uda_loader,
                    max_examples_per_epoch=max_examples_per_epoch)

            # Log the metrics for this epoch
            metrics = ray_evaluate(model, eval_loaders)
            metrics.update(dict(train_loss=train_loss))
            checkpoint = get_ray_checkpoint(model)
            session.report(metrics, checkpoint=checkpoint)

    # Get the default/fixed configs (these are provided to every Trainer but
    # can be overwritten if they are also in the param_space).
    default_train_config = get_default_config(model_name, dset)

    # Construct the Trainer object that will be passed to each worker.
    if is_pytorch_model_name(model_name):

        split_by_domain = is_domain_generalization_model_name(model_name)
        datasets = prepare_ray_datasets(dset, split_by_domain,
                                        prepare_pytorch=True)

        use_gpu = torch.cuda.is_available()
        trainer = TorchTrainer(
            train_loop_per_worker=train_loop_per_worker,
            train_loop_config=default_train_config,
            datasets=datasets,
            scaling_config=ScalingConfig(
                num_workers=tune_config.num_workers,
                resources_per_worker={
                    "GPU": tune_config.gpu_per_worker if use_gpu else 0,
                    "CPU": tune_config.cpu_per_worker},
                _max_cpu_fraction_per_node=0.8,
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
        datasets = prepare_ray_datasets(dset,
                                        split_train_loaders_by_domain=False,
                                        prepare_pytorch=False)
        scaling_config = ScalingConfig(
            num_workers=tune_config.num_workers,
            # Set trainer_resources as described in
            # https://docs.ray.io/en/latest/train/gbdt.html#how-to-scale-out-training
            trainer_resources={"CPU": 0},
            use_gpu=False,
            resources_per_worker={"CPU": tune_config.cpu_per_worker},
            _max_cpu_fraction_per_node=0.8)
        params = {
            # Note: tree_method must be gpu_hist if using GPU.
            "tree_method": "hist",
            "objective": "binary:logistic",
            # Note: the `map` in sklearn is *not* directly comparable
            # to the average precision score for lightgbm and sklearn.
            "eval_metric": ["error", "auc", "map"]}
        trainer = XGBoostTrainer(label_column=dset.target,
                                 datasets=datasets,
                                 params=params,
                                 scaling_config=scaling_config)
        param_space = {"params": search_space[model_name]}

    elif model_name == "lightgbm":
        scaling_config = ScalingConfig(
            num_workers=tune_config.num_workers,
            # Set trainer_resources as described in
            # https://docs.ray.io/en/latest/train/gbdt.html#how-to-scale-out-training
            trainer_resources={"CPU": 0},
            use_gpu=False,
            resources_per_worker={"CPU": tune_config.cpu_per_worker},
            _max_cpu_fraction_per_node=0.8)
        datasets = prepare_ray_datasets(dset,
                                        split_train_loaders_by_domain=False,
                                        prepare_pytorch=False)
        params = {"objective": "binary",
                  # Note that for lightgbm, average_precision <=> sklearn's
                  # average_precision_score.
                  "metric": ["binary_error", "auc", "average_precision"],
                  # use only the first metric for early stopping;
                  # the others are only for evaluation.
                  "first_metric_only": True,
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
                             local_dir=tune_config.ray_local_dir),
        param_space=param_space,
        tune_config=tune.TuneConfig(
            search_alg=tune_config.get_search_alg(),
            scheduler=tune_config.get_scheduler(),
            num_samples=tune_config.num_samples,
            time_budget_s=tune_config.time_budget_hrs * 3600 if tune_config.time_budget_hrs else None,
            max_concurrent_trials=tune_config.max_concurrent_trials))

    results = tuner.fit()
    ray.shutdown()
    auto_garbage_collect(force=True)
    try:
        cmd = "kill -9 $(lsof +L1 /dev/shm | grep deleted | awk '{print $2}')"
        print(f"[INFO] attempting to clean up files with {cmd}")
        os.system(cmd)
    except Exception as e:
        print(
            f"[WARNING] exception running cleanup: {e}. Suggest running this command manually (due to Ray bug): {cmd}")
    return results


def fetch_postprocessed_results_df(
        results: ray.tune.ResultGrid) -> pd.DataFrame:
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

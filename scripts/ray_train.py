import argparse
from typing import Dict
from ray import tune
from ray.tune import Tuner
from ray.tune.schedulers import ASHAScheduler
from ray.tune.search.hyperopt import HyperOptSearch
from ray.air.config import RunConfig
from ray.train.torch import TorchTrainer
from ray.train.xgboost import XGBoostTrainer
from ray.train.lightgbm import LightGBMTrainer
from ray.air.config import ScalingConfig
from ray import train
from ray.air import session
import torch

from tablebench.core import TabularDataset, TabularDatasetConfig
from tablebench.datasets.experiment_configs import EXPERIMENT_CONFIGS
from tablebench.models.utils import get_estimator
from tablebench.models.config import get_default_config
from tablebench.models.compat import SklearnStylePytorchModel, \
    is_pytorch_model_name
from tablebench.models.training import get_optimizer, train_epoch
from tablebench.configs.hparams import search_space
from tablebench.models.expgrad import ExponentiatedGradientTrainer
from tablebench.models.ray_utils import prepare_torch_datasets, ray_evaluate, \
    get_ray_checkpoint, make_ray_dataset


def main(experiment: str, model_name: str, cache_dir: str,
         debug: bool,
         no_tune: bool, num_samples: int,
         tune_metric_name: str = "validation_accuracy",
         tune_metric_higher_is_better: bool = True,
         max_concurrent_trials=2,
         num_workers=1,
         early_stop=True, max_epochs=100):
    if debug:
        print("[INFO] running in debug mode.")
        experiment = "_debug"
        num_samples = 1

    expt_config = EXPERIMENT_CONFIGS[experiment]

    dataset_config = TabularDatasetConfig(cache_dir=cache_dir)
    tabular_dataset_kwargs = expt_config.tabular_dataset_kwargs
    if "name" not in tabular_dataset_kwargs:
        tabular_dataset_kwargs["name"] = experiment

    dset = TabularDataset(config=dataset_config,
                          splitter=expt_config.splitter,
                          grouper=expt_config.grouper,
                          preprocessor_config=expt_config.preprocessor_config,
                          **tabular_dataset_kwargs)

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

        n_epochs = config["n_epochs"] if not early_stop else max_epochs
        device = train.torch.get_device()
        for epoch in range(max_epochs):
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
        num_workers=num_workers,
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
        trainer = XGBoostTrainer(label_column=dset.target,
                                 datasets=datasets,
                                 params={"tree_method": "hist",
                                         "objective": "binary:logistic",
                                         "eval_metric": "error"},
                                 scaling_config=scaling_config)
        tune_metric_name = "validation-error"
        tune_metric_higher_is_better = False
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
        tune_metric_name = "validation-binary_error"
        tune_metric_higher_is_better = False

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

    if no_tune:
        # To run just a single training iteration (without tuning)
        result = trainer.fit()
        latest_checkpoint = result.checkpoint
        return

    # Create Tuner.

    mode = "max" if tune_metric_higher_is_better else "min"

    stopper = tune.stopper.ExperimentPlateauStopper(
        metric=tune_metric_name,
        mode=mode,
        # TODO(jpgard): increase patience to 16 as in
        #  https://arxiv.org/pdf/2106.11959.pdf
        patience=5)

    tuner = Tuner(
        trainable=trainer,
        run_config=RunConfig(name="test_tuner_notebook",
                             local_dir="ray-results",
                             stop=stopper),
        param_space=param_space,
        tune_config=tune.TuneConfig(
            search_alg=HyperOptSearch(metric=tune_metric_name, mode=mode),
            scheduler=ASHAScheduler(
                time_attr='training_iteration',
                metric=tune_metric_name,
                mode=mode,
                stop_last_trials=True),
            num_samples=num_samples,
            max_concurrent_trials=max_concurrent_trials))

    results = tuner.fit()

    results_df = results.get_dataframe()
    print(results_df)
    results_df.to_csv(f"tune_results_{experiment}_{model_name}.csv",
                      index=False)
    return results_df


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache_dir", default="tmp",
                        help="Directory to cache raw data files to.")
    parser.add_argument("--debug", action="store_true", default=False,
                        help="Whether to run in debug mode. If True, various "
                             "truncations/simplifications are performed to "
                             "speed up experiment.")
    parser.add_argument("--experiment", default="adult",
                        help="Experiment to run. Overridden when debug=True.")
    parser.add_argument("--model_name", default="mlp")
    parser.add_argument("--num_samples", type=int, default=1,
                        help="Number of hparam samples to take in tuning "
                             "sweep.")
    parser.add_argument("--no_tune", action="store_true", default=False,
                        help="If set, suppresses hyperparameter tuning of the "
                             "model (for faster testing).")
    args = parser.parse_args()
    main(**vars(args))

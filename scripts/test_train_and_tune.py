import argparse

from ray import air, tune

from tablebench.configs.hparams import search_space
from tablebench.core import TabularDataset, TabularDatasetConfig
from tablebench.datasets.experiment_configs import EXPERIMENT_CONFIGS
from tablebench.models import get_estimator, get_model_config
from tablebench.models.training import train


def main(experiment: str, device: str, model: str, cache_dir: str, debug: bool,
         no_tune: bool, num_samples: int, tune_metric_name="metric",
         tune_metric_higher_is_better: bool = True):
    if debug:
        print("[INFO] running in debug mode.")
        experiment = "_debug"

    if experiment not in EXPERIMENT_CONFIGS:
        raise NotImplementedError(f"{experiment} is not implemented.")

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

    def _train_fn(run_config=None):
        # Get the default configs
        config = get_model_config(model, dset)
        if run_config:
            # Override the defaults with run_config, if provided.
            config.update(run_config)
        estimator = get_estimator(model, **config)
        train(estimator, dset, device=device, config=config,
              tune_report_split="ood_test")

    if no_tune:
        _train_fn()
    else:
        tuner = tune.Tuner(
            _train_fn,
            param_space=search_space[model],
            tune_config=tune.tune_config.TuneConfig(num_samples=num_samples),
            run_config=air.RunConfig(local_dir="./ray-results",
                                     name="test_experiment"))

        results = tuner.fit()

        best_result = results.get_best_result(
            tune_metric_name, "max" if tune_metric_higher_is_better else "min")

        print("Best trial config: {}".format(best_result.config))
        print("Best trial final {}: {}".format(
            tune_metric_name,
            best_result.metrics[tune_metric_name]))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache_dir", default="tmp",
                        help="Directory to cache raw data files to.")
    parser.add_argument("--debug", action="store_true", default=False,
                        help="Whether to run in debug mode. If True, various "
                             "truncations/simplifications are performed to "
                             "speed up experiment.")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--experiment", default="adult",
                        help="Experiment to run. Overridden when debug=True.")
    parser.add_argument("--model", default="mlp")
    parser.add_argument("--num_samples", type=int, default=1,
                        help="Number of hparam samples to take in tuning "
                             "sweep.")
    parser.add_argument("--no_tune", action="store_true", default=False,
                        help="If set, suppresses hyperparameter tuning of the "
                             "model (for faster testing).")
    args = parser.parse_args()
    main(**vars(args))

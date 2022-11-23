import argparse

from ray import air, tune

from tablebench.core import TabularDataset, TabularDatasetConfig
from tablebench.datasets.experiment_configs import EXPERIMENT_CONFIGS
from tablebench.models import get_estimator, get_model_config
from tablebench.models.training import train

search_space = {
    "d_hidden": tune.choice([64, 128, 256, 512]),

    # Samples a float uniformly between 0.0001 and 0.1, while
    # sampling in log space and rounding to multiples of 0.00005
    "lr": tune.qloguniform(1e-4, 1e-1, 5e-5),

    "n_epochs": tune.randint(1, 2),
    "num_layers": tune.randint(1, 4),
    "weight_decay": tune.quniform(0., 1., 0.1),
}


def main(experiment: str, device: str, model: str, cache_dir: str, debug: bool):
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

    def _train_fn(run_config):
        # Get the default configs
        config = get_model_config(model, dset)
        # Override the defaults with run_config
        config.update(run_config)
        estimator = get_estimator(model, **config)
        train(estimator, dset, device=device, config=config)

    tuner = tune.Tuner(
        _train_fn,
        param_space=search_space,
        tune_config=tune.tune_config.TuneConfig(num_samples=1),
        run_config=air.RunConfig(local_dir="./ray-results",
                                 name="test_experiment")
    )

    tuner.fit()


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
    parser.add_argument("--model", default="mlp", choices=(
        "ft_transformer", "mlp", "resnet", "group_dro"))
    args = parser.parse_args()
    main(**vars(args))

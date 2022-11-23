import argparse
from tablebench.core import TabularDataset, TabularDatasetConfig

from tablebench.datasets.experiment_configs import EXPERIMENT_CONFIGS
from tablebench.models import get_estimator
from tablebench.models.training import train


def main(experiment, cache_dir, model, debug: bool):
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
    estimator = get_estimator(model)
    train(estimator, dset)
    return


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache_dir", default="tmp",
                        help="Directory to cache raw data files to.")
    parser.add_argument("--debug", action="store_true", default=False,
                        help="Whether to run in debug mode. If True, various "
                             "truncations/simplifications are performed to "
                             "speed up experiment.")
    parser.add_argument("--experiment", choices=list(EXPERIMENT_CONFIGS.keys()),
                        default="brfss_diabetes_st",
                        help="Experiment to run. Overridden when debug=True.")
    parser.add_argument("--model", default="histgbm",
                        help="model to use.")
    args = parser.parse_args()
    main(**vars(args))

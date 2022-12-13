"""Cache a tabualr dataset to parquet."""
import argparse

from tablebench.core import TabularDataset, TabularDatasetConfig
from tablebench.datasets.experiment_configs import EXPERIMENT_CONFIGS


def main(cache_dir, experiment):
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

    dset.to_sharded()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache_dir", default="tmp",
                        help="Directory to cache raw data files to.")
    parser.add_argument("--experiment", default="adult",
                        help="Experiment to run. Overridden when debug=True.")

    args = parser.parse_args()
    main(**vars(args))

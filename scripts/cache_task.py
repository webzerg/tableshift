"""Cache a tabualr dataset to parquet."""
import argparse

from tablebench.core import TabularDataset, TabularDatasetConfig
from tablebench.datasets.experiment_configs import EXPERIMENT_CONFIGS, ExperimentConfig
from tablebench.configs.domain_shift import domain_shift_experiment_configs


def _cache_experiment(expt_config: ExperimentConfig, cache_dir, overwrite: bool):
    dataset_config = TabularDatasetConfig(cache_dir=cache_dir)
    tabular_dataset_kwargs = expt_config.tabular_dataset_kwargs
    if "name" not in tabular_dataset_kwargs:
        tabular_dataset_kwargs["name"] = experiment

    dset = TabularDataset(config=dataset_config,
                          splitter=expt_config.splitter,
                          grouper=expt_config.grouper,
                          preprocessor_config=expt_config.preprocessor_config,
                          initialize_data=False,
                          **tabular_dataset_kwargs)
    if dset.is_cached() and (not overwrite):
        print(f"dataset with uid {dset._get_uid()} is already cached; skipping")

    else:
        dset._initialize_data()
        dset.to_sharded()
    return


def main(cache_dir, experiment, overwrite: bool, domain_shift_experiment=None):
    assert not (experiment and domain_shift_experiment), "specify either experiment or domain_shift_experiment, " \
                                                         "but not both. "
    if experiment:
        expt_config = EXPERIMENT_CONFIGS[experiment]
        _cache_experiment(expt_config, cache_dir, overwrite=overwrite)

    else:
        assert domain_shift_experiment

    domain_shift_expt_config = domain_shift_experiment_configs[domain_shift_experiment]

    for expt_config in domain_shift_expt_config.as_experiment_config_iterator():
        try:
            _cache_experiment(expt_config, cache_dir, overwrite=overwrite)
        except Exception as e:
            print(f"exception when caching experiment with ood values {expt_config.splitter.domain_split_ood_values}: "
                  f"{e}")
            continue


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache_dir", default="tmp",
                        help="Directory to cache raw data files to.")
    parser.add_argument("--domain_shift_experiment", "-d",
                        help="Experiment to run. Overridden when debug=True."
                             "Example value: 'diabetes_admtype'.")
    parser.add_argument("--experiment",
                        help="Experiment to run. Overridden when debug=True."
                             "Example value: 'adult'.")
    parser.add_argument("--overwrite", action="store_true", default=False)

    args = parser.parse_args()
    main(**vars(args))

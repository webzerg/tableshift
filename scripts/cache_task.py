"""Cache a tabualr dataset to parquet."""
import argparse
import logging

from tablebench.core import TabularDataset, TabularDatasetConfig
from tablebench.configs.experiment_configs import EXPERIMENT_CONFIGS, \
    ExperimentConfig
from tablebench.configs.domain_shift import domain_shift_experiment_configs
from tablebench.core.utils import make_uid

LOG_LEVEL = logging.DEBUG

logger = logging.getLogger()
logging.basicConfig(
    format='%(asctime)s %(levelname)-8s [%(filename)s:%(lineno)d] %(message)s',
    level=LOG_LEVEL,
    datefmt='%Y-%m-%d %H:%M:%S')


def _cache_experiment(expt_config: ExperimentConfig, cache_dir,
                      overwrite: bool,
                      no_domains_to_subdirectories: bool):
    dataset_config = TabularDatasetConfig(cache_dir=cache_dir)
    tabular_dataset_kwargs = expt_config.tabular_dataset_kwargs
    assert "name" in tabular_dataset_kwargs

    dset = TabularDataset(config=dataset_config,
                          splitter=expt_config.splitter,
                          grouper=expt_config.grouper,
                          preprocessor_config=expt_config.preprocessor_config,
                          initialize_data=False,
                          **tabular_dataset_kwargs)
    if dset.is_cached() and (not overwrite):
        uid = make_uid(tabular_dataset_kwargs["name"], expt_config.splitter)
        logging.info(f"dataset with uid {uid} is already cached; skipping")

    else:
        domains_to_subdirectories = not no_domains_to_subdirectories
        logging.info(
            f"domains_to_subdirectories is {domains_to_subdirectories}")
        dset._initialize_data()
        dset.to_sharded(domains_to_subdirectories=domains_to_subdirectories)
    return


def main(cache_dir,
         experiment,
         overwrite: bool,
         no_domains_to_subdirectories: bool = False,
         domain_shift_experiment=None):
    assert (experiment or domain_shift_experiment) and \
           not (experiment and domain_shift_experiment), \
        "specify either experiment or domain_shift_experiment, but not both."

    cache_kwargs = {
        "cache_dir": cache_dir,
        "overwrite": overwrite,
        "no_domains_to_subdirectories": no_domains_to_subdirectories,
    }
    logging.debug(f"cache_kwargs is: {cache_kwargs}")
    if experiment:
        expt_config = EXPERIMENT_CONFIGS[experiment]
        _cache_experiment(expt_config, **cache_kwargs)
        print("caching tasks complete!")
        return

    domain_shift_expt_config = domain_shift_experiment_configs[
        domain_shift_experiment]

    for expt_config in domain_shift_expt_config.as_experiment_config_iterator():
        try:
            _cache_experiment(expt_config, **cache_kwargs)
        except Exception as e:
            print(f"exception when caching experiment with ood values "
                  f"{expt_config.splitter.domain_split_ood_values}: {e}")
            continue
    print("caching tasks complete!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache_dir", default="tmp",
                        help="Directory to cache raw data files to.")
    parser.add_argument("--domain_shift_experiment", "-d",
                        help="Experiment to run. Overridden when debug=True."
                             "Example value: 'physionet_set'.")
    parser.add_argument("--no_domains_to_subdirectories",
                        action="store_true",
                        help="If set, domains will NOT be written to separate"
                             "subdirectories. For example, instead of writing files to"
                             "/train/1/train_1.csv where the second level is "
                             "the domain value, they will be written to "
                             "/train/train_1.csv and not split by the domain"
                             "value. Useful when using thresholding.")
    parser.add_argument("--experiment",
                        help="Experiment to run. Overridden when debug=True."
                             "Example value: 'adult'.")
    parser.add_argument("--overwrite", action="store_true", default=False)

    args = parser.parse_args()
    main(**vars(args))

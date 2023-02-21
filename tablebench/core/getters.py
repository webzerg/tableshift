from tablebench.configs.experiment_configs import EXPERIMENT_CONFIGS
from .tabular_dataset import TabularDataset, TabularDatasetConfig


def get_dataset(name: str, cache_dir: str = "tmp") -> TabularDataset:
    assert name in EXPERIMENT_CONFIGS.keys(), \
        f"Dataset name {name} is not available; choices are: " \
        f"{sorted(EXPERIMENT_CONFIGS.keys())}"

    expt_config = EXPERIMENT_CONFIGS[name]
    dataset_config = TabularDatasetConfig(cache_dir=cache_dir)
    tabular_dataset_kwargs = expt_config.tabular_dataset_kwargs
    if "name" not in tabular_dataset_kwargs:
        tabular_dataset_kwargs["name"] = name

    dset = TabularDataset(config=dataset_config,
                          splitter=expt_config.splitter,
                          grouper=expt_config.grouper,
                          preprocessor_config=expt_config.preprocessor_config,
                          **tabular_dataset_kwargs)
    return dset

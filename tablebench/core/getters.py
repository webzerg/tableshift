from typing import Optional

from tablebench.configs.experiment_configs import EXPERIMENT_CONFIGS
from .tabular_dataset import TabularDataset, TabularDatasetConfig
from .features import PreprocessorConfig


def get_dataset(name: str, cache_dir: str = "tmp",
                preprocessor_config: Optional[
                    PreprocessorConfig] = None) -> TabularDataset:
    """Helper function to fetch a dataset with the default benchmark parameters.

    Args:
        name: the dataset name.
        cache_dir: the cache directory to use. TableShift will check for cached
            data files here before downloading.
        preprocessor_config: optional Preprocessor to override the default
            preprocessor config. If using the TableShift benchmark, it is
            recommended to leave this as None to use the default preprocessor.
        """
    assert name in EXPERIMENT_CONFIGS.keys(), \
        f"Dataset name {name} is not available; choices are: " \
        f"{sorted(EXPERIMENT_CONFIGS.keys())}"

    expt_config = EXPERIMENT_CONFIGS[name]
    dataset_config = TabularDatasetConfig(cache_dir=cache_dir)
    tabular_dataset_kwargs = expt_config.tabular_dataset_kwargs
    if "name" not in tabular_dataset_kwargs:
        tabular_dataset_kwargs["name"] = name

    if preprocessor_config is None:
        preprocessor_config = expt_config.preprocessor_config
    dset = TabularDataset(
        config=dataset_config,
        splitter=expt_config.splitter,
        grouper=expt_config.grouper,
        preprocessor_config=preprocessor_config,
        **tabular_dataset_kwargs)
    return dset

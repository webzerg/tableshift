"""Data sources for TableBench."""
from abc import ABC, abstractmethod
from functools import partial
import os
import subprocess
from typing import Sequence, Callable, Optional
import zipfile

import pandas as pd

from . import utils
from .features import FeatureList
from tablebench.datasets import *


class DataSource(ABC):
    """Abstract class to represent a generic data source."""

    def __init__(self, cache_dir: str,
                 preprocess_fn: Callable[[pd.DataFrame], pd.DataFrame],
                 resources: Sequence[str] = None,
                 download: bool = True,
                 feature_list: Optional[FeatureList] = None
                 ):
        self.cache_dir = cache_dir
        self.download = download
        # The feature_list describes the schema of the data *after* the
        # preprocess_fn is applied. It is used to check the output of the
        # preprocess_fn, and features are dropped or type-cast as necessary.
        self.feature_list = feature_list
        self.preprocess_fn = preprocess_fn
        self.resources = resources
        self._initialize_cache_dir()

    def _initialize_cache_dir(self):
        """Create cache_dir if it does not exist."""
        utils.initialize_dir(self.cache_dir)

    def get_data(self) -> pd.DataFrame:
        """Fetch data from local cache or download if necessary."""
        self._download_if_not_cached()
        raw_data = self._load_data()
        return self.preprocess_fn(raw_data)

    def _download_if_not_cached(self):
        """Download files if they are not already cached."""
        for url in self.resources:
            utils.download_file(url, self.cache_dir)

    @abstractmethod
    def _load_data(self) -> pd.DataFrame:
        """Load the raw data from disk and return it.

        Any preprocessing should be performed in preprocess_fn, not here."""
        raise

    @property
    def is_cached(self) -> bool:
        """Check whether all resources exist in cache dir."""
        for url in self.resources:
            basename = utils.basename_from_url(url)
            fp = os.path.join(self.cache_dir, basename)
            if not os.path.exists(fp):
                return False
        return True


class KaggleDataSource(DataSource):
    def __init__(
            self,
            kaggle_dataset_name: str,
            kaggle_creds_dir="~/.kaggle",
            **kwargs):
        self.kaggle_creds_dir = kaggle_creds_dir
        self.kaggle_dataset_name = kaggle_dataset_name
        super().__init__(**kwargs)

    @property
    def kaggle_creds(self):
        return os.path.expanduser(
            os.path.join(self.kaggle_creds_dir, "kaggle.json"))

    @abstractmethod
    def _load_data(self) -> pd.DataFrame:
        # Should be implemented by subclasses.
        raise

    def _download_kaggle_data(self):
        """Download the data from Kaggle."""
        # Check Kaggle authentication.
        assert os.path.exists(self.kaggle_creds), \
            f"No kaggle credentials found at {self.kaggle_creds}."
        "Create an access token at https://www.kaggle.com/YOUR_USERNAME/account"
        f"and store it at {self.kaggle_creds}."

        # Download the dataset using Kaggle CLI.
        cmd = "kaggle datasets download " \
              f"-d {self.kaggle_dataset_name} " \
              f"-p {self.cache_dir}"
        print(f"running {cmd}")
        res = subprocess.run(cmd, shell=True)
        print(f"{cmd} returned {res}")
        return

    @abstractmethod
    def _download_if_not_cached(self):
        # Should be implemnented by subclasses.
        raise


class BRFSSDataSource(KaggleDataSource):
    def __init__(
            self,
            kaggle_dataset_name="cdc/behavioral-risk-factor-surveillance-system",
            preprocess_fn=preprocess_brfss,
            years=(2015,),
            **kwargs):
        self.years = years  # Which years to use BRFSS survey data from.
        super(BRFSSDataSource, self).__init__(
            kaggle_dataset_name=kaggle_dataset_name,
            preprocess_fn=preprocess_fn,
            **kwargs)

    def _download_if_not_cached(self):
        self._download_kaggle_data()
        # location of the local zip file
        zip_fp = os.path.join(self.cache_dir,
                              "behavioral-risk-factor-surveillance-system.zip")
        # where to unzip the file to
        unzip_dest = os.path.join(self.cache_dir, self.kaggle_dataset_name)
        with zipfile.ZipFile(zip_fp, 'r') as zf:
            zf.extractall(unzip_dest)

    def _load_data(self) -> pd.DataFrame:
        df_list = []
        for year in self.years:
            fp = os.path.join(self.cache_dir,
                              self.kaggle_dataset_name,
                              f"{year}.csv")
            df = pd.read_csv(fp, usecols=BRFSS_INPUT_FEATURES)
            df_list.append(df)
        return pd.concat(df_list, axis=0)


class ACSDataSource(DataSource):
    def __init__(self,
                 preprocess_fn=preprocess_acsincome,
                 task="acsincome",
                 year: int = 2018,
                 states=ACS_STATE_LIST,
                 feature_mapping="coarse",
                 **kwargs):
        self.acs_task = task.lower().replace("acs", "")
        self.feature_mapping = get_feature_mapping(feature_mapping)
        self.states = states
        self.year = year
        self.acs_data = None  # holds the cached data from folktables source.
        super().__init__(preprocess_fn=preprocess_fn, **kwargs)

    def _get_acs_data(self):
        if self.acs_data is None:
            print("fetching ACS data...")
            data_source = get_acs_data_source(self.year, self.cache_dir)
            self.acs_data = data_source.get_data(states=self.states,
                                                 download=True)
            print("fetching ACS data complete.")
        else:
            print("fetching cached ACS data.")
        return self.acs_data

    def _download_if_not_cached(self):
        if self.acs_data is None:
            return self._get_acs_data()
        else:
            return self.acs_data

    def _load_data(self) -> pd.DataFrame:
        acs_data = self._get_acs_data()
        task_config = ACS_TASK_CONFIGS[self.acs_task]
        target_transform = partial(task_config.target_transform,
                                   threshold=task_config.threshold)
        ACSProblem = folktables.BasicProblem(
            features=task_config.features_to_use.names,
            target=task_config.target,
            target_transform=target_transform,
            preprocess=task_config.preprocess,
            postprocess=task_config.postprocess,
        )
        X, y, _ = ACSProblem.df_to_numpy(acs_data)
        df = acs_data_to_df(X, y, task_config.features_to_use.names,
                            feature_mapping=self.feature_mapping)
        return df


class AdultDataSource(DataSource):
    """Data source for the Adult dataset."""

    def __init__(self, resources=ADULT_RESOURCES,
                 preprocess_fn=preprocess_adult, **kwargs):
        super().__init__(resources=resources,
                         preprocess_fn=preprocess_fn, **kwargs)

    def _load_data(self) -> pd.DataFrame:
        train_fp = os.path.join(self.cache_dir, "adult.data")
        train = pd.read_csv(
            train_fp,
            names=ADULT_FEATURE_NAMES,
            sep=r'\s*,\s*',
            engine='python', na_values="?")
        train["Split"] = "train"

        test_fp = os.path.join(self.cache_dir, "adult.test")

        test = pd.read_csv(
            test_fp,
            names=ADULT_FEATURE_NAMES,
            sep=r'\s*,\s*',
            engine='python', na_values="?", skiprows=1)
        test["Split"] = "test"

        return pd.concat((train, test))


class COMPASDataSource(DataSource):
    def __init__(self, resources=COMPAS_RESOURCES,
                 preprocess_fn=preprocess_compas, **kwargs):
        super().__init__(resources=resources,
                         preprocess_fn=preprocess_fn, **kwargs)

    def _load_data(self) -> pd.DataFrame:
        df = pd.read_csv(
            os.path.join(self.cache_dir, "compas-scores-two-years.csv"))
        return df


class GermanDataSource(DataSource):
    def __init__(self, resources=GERMAN_RESOURCES,
                 preprocess_fn=preprocess_german, **kwargs):
        super().__init__(resources=resources, preprocess_fn=preprocess_fn,
                         **kwargs)

    def _load_data(self) -> pd.DataFrame:
        df = pd.read_csv(os.path.join(self.cache_dir, "german.data"),
                         sep=" ", header=None)
        return df


# Mapping of dataset names to their DataSource classes.
_DATA_SOURCE_CLS = {
    "acsincome": ACSDataSource,
    "brfss": BRFSSDataSource,
    "adult": AdultDataSource,
    "compas": COMPASDataSource,
    "german": GermanDataSource,
}


def get_data_source(name: str, **kwargs) -> DataSource:
    if name in _DATA_SOURCE_CLS:
        cls = _DATA_SOURCE_CLS[name]
        return cls(**kwargs)
    else:
        raise NotImplementedError(f"data source {name} not implemented.")

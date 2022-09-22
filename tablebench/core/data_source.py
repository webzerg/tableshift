"""Data sources for TableBench."""
from abc import ABC, abstractmethod
import os
from typing import Sequence

import pandas as pd

from . import utils
from tablebench.datasets import *


class DataSource(ABC):
    """Abstract class to represent a generic data source."""

    def __init__(self, cache_dir: str, resources: Sequence[str],
                 download: bool = True):
        self.cache_dir = cache_dir
        self.download = download
        self.resources = resources
        self._initialize_cache_dir()
        self.get_data()

    def _initialize_cache_dir(self):
        """Create cache_dir if it does not exist."""
        utils.initialize_dir(self.cache_dir)

    def get_data(self) -> pd.DataFrame:
        """Fetch data from local cache or download if necessary."""
        self._download_if_not_cached()
        return self._load_data()

    @abstractmethod
    def _download_if_not_cached(self):
        """Download the dataset."""
        raise

    @abstractmethod
    def _load_data(self) -> pd.DataFrame:
        """Load the data from disk and return it."""
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


class UCIDataSource(DataSource):

    def _download_if_not_cached(self):
        # Download files if they are not already cached
        for url in self.resources:
            utils.download_file(url, self.cache_dir)


class AdultDataSource(UCIDataSource):
    """Data source for the Adult dataset."""

    def __init__(self, cache_dir: str, resources=ADULT_RESOURCES,
                 download: bool = True):
        super().__init__(cache_dir=cache_dir, resources=resources,
                         download=download)

    def _load_data(self):
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


def get_data_source(name: str, **kwargs) -> DataSource:
    if name == "adult":
        return AdultDataSource(**kwargs)
    else:
        raise NotImplementedError

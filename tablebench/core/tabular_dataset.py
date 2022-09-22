from abc import ABC
from dataclasses import dataclass
from typing import Optional, Tuple, Callable

import pandas as pd

from .splitter import Splitter, DomainSplitter
from .grouper import Grouper
from .data_source import get_data_source
from .features import FeatureList, PreprocessorConfig
from tablebench.datasets import _DEFAULT_FEATURES, _PREPROCESS_FNS


@dataclass
class TabularDatasetConfig:
    cache_dir: str
    download: bool
    random_seed: int


class TabularDataset(ABC):
    def __init__(self, name: str, config: TabularDatasetConfig,
                 splitter: Splitter,
                 preprocessor_config: PreprocessorConfig,
                 grouper: Optional[Grouper],
                 feature_list: Optional[FeatureList] = None):
        self.name = name
        self.config = config
        self.grouper = grouper
        self.preprocessor_config = preprocessor_config
        self.splitter = splitter

        # Dataset-specific info: features, data source, preprocessing.

        self.feature_list = feature_list if feature_list else _DEFAULT_FEATURES[
            self.name]
        self.data_source = get_data_source(name=self.name,
                                           cache_dir=self.config.cache_dir,
                                           download=self.config.download)

        # Placeholders for data/labels/groups and split indices.
        self.data = None
        self.labels = None
        self.groups = None
        self.splits = None  # dict mapping {split_name: list of idxs in split}

        self._initialize_data()

    @property
    def features(self):
        return self.feature_list.names

    def _initialize_data(self):
        """Load the data/labels/groups from a data source."""
        data = self.data_source.get_data()
        data = self._process_pre_split(data)
        self._generate_splits(data)
        if "Split" in data.columns:
            data.drop(columns=["Split"], inplace=True)
        data = self._process_post_split(data)

        X, y, G = self._X_y_G_split(data)
        self.data = X
        self.labels = y
        self.groups = G
        del data
        return

    def _X_y_G_split(self, data) -> \
            Tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
        """Fetch the (data, labels, groups) arrays from a DataFrame."""
        data_features = [x for x in data.columns
                         if x not in self.grouper.features + ["Target"]]
        if not self.grouper.drop:
            # Retain the group variables as features.
            data_features.extend(self.grouper.features)

        X = data.loc[:, data_features]
        y = data.loc[:, "Target"]
        G = data.loc[:, self.grouper.features]
        return X, y, G

    def _process_pre_split(self, data: pd.DataFrame) -> pd.DataFrame:
        """Dataset-specific preprocessing function.

        Conducts any preprocessing needed **before** splitting
        (i.e. feature selection, filtering, etc.)."""
        cols = list(set(self.features + self.grouper.features + ["Target"]))
        if "Split" in data.columns:
            cols.append("Split")
        return data.loc[:, cols]

    def _generate_splits(self, data):
        """Call the splitter to generate splits for the dataset."""
        assert self.splits is None, "attempted to overwrite existing splits."

        data, labels, groups = self._X_y_G_split(data)

        self.splits = self.splitter(data, labels, groups)
        return

    def _post_split_feature_selection(self,
                                      data: pd.DataFrame) -> pd.DataFrame:
        """Select features for post-split processing."""
        if isinstance(self.splitter, DomainSplitter):
            # Case: domain split; now that the split has been made, drop the
            # domain split feature.
            return data.drop(columns=self.splitter.domain_split_varname)
        else:
            return data

    def _process_post_split(self, data) -> pd.DataFrame:
        """Dataset-specific postprocessing function.

        Conducts any processing required **after** splitting (e.g.
        normalization, drop features needed only for splitting)."""
        data = self._post_split_feature_selection(data)
        data = self.grouper.transform(data)
        data = self.preprocessor_config.fit_transform(
            data,
            self.splits["train"],
            passthrough_columns=self.grouper.features)
        return data

    def get_pandas(self, split) -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
        """Fetch the (data, labels, groups) for this TabularDataset."""
        assert split in self.splits.keys(), \
            f"split {split} not in {list(self.splits.keys)}"
        idxs = self.splits[split]
        return (self.data.iloc[idxs],
                self.labels.iloc[idxs],
                self.groups.iloc[idxs])

    def get_dataloader(self):
        raise NotImplementedError

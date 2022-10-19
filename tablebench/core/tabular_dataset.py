from abc import ABC
from dataclasses import dataclass
from typing import Optional, Tuple, Callable

import pandas as pd

from .splitter import Splitter, DomainSplitter
from .grouper import Grouper
from .tasks import get_task_config
from .features import PreprocessorConfig


@dataclass
class TabularDatasetConfig:
    cache_dir: str = "tmp"
    download: bool = True
    random_seed: int = 948324


class TabularDataset(ABC):
    def __init__(self, name: str, config: TabularDatasetConfig,
                 splitter: Splitter,
                 preprocessor_config: PreprocessorConfig,
                 grouper: Optional[Grouper],
                 **kwargs):
        self.name = name
        self.config = config
        self.grouper = grouper
        self.preprocessor_config = preprocessor_config
        self.splitter = splitter

        # Dataset-specific info: features, data source, preprocessing.

        self.task_config = get_task_config(self.name)
        self.data_source = self.task_config.data_source_cls(
            cache_dir=self.config.cache_dir,
            download=self.config.download,
            **kwargs)

        # Placeholders for data/labels/groups and split indices.
        self.data = None
        self.labels = None
        self.groups = None
        self.splits = None  # dict mapping {split_name: list of idxs in split}

        self._initialize_data()

    @property
    def features(self):
        return self.task_config.feature_list.names

    @property
    def predictors(self):
        return self.task_config.feature_list.predictors

    @property
    def target(self):
        return self.task_config.feature_list.target

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

    def _X_y_G_split(self, data, default_targets_dtype=int) -> \
            Tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
        """Fetch the (data, labels, groups) arrays from a DataFrame."""
        data_features = [x for x in data.columns
                         if x not in self.grouper.features + [self.target]]
        if not self.grouper.drop:
            # Retain the group variables as features.
            data_features.extend(self.grouper.features)

        X = data.loc[:, data_features]
        y = data.loc[:, self.target]
        G = data.loc[:, self.grouper.features]

        if y.dtype == "O":
            y = y.astype(default_targets_dtype)
        return X, y, G

    def _process_pre_split(self, data: pd.DataFrame) -> pd.DataFrame:
        """Dataset-specific preprocessing function.

        Conducts any preprocessing needed **before** splitting
        (i.e. feature selection, filtering, grouping etc.)."""
        cols = list(set(self.predictors +
                        self.grouper.features +
                        [self.target]))
        if "Split" in data.columns:
            cols.append("Split")
        data = self.grouper.transform(data)
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
            data.drop(columns=self.splitter.domain_split_varname, inplace=True)

        data = self.task_config.feature_list.apply_schema(data)
        return data

    def _process_post_split(self, data) -> pd.DataFrame:
        """Dataset-specific postprocessing function.

        Conducts any processing required **after** splitting (e.g.
        normalization, drop features needed only for splitting)."""
        data = self._post_split_feature_selection(data)
        data = self.preprocessor_config.fit_transform(
            data,
            self.splits["train"],
            passthrough_columns=self.grouper.features + [self.target])
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

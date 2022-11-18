from abc import ABC
from dataclasses import dataclass
from typing import Optional, Tuple, Union

import numpy as np
import pandas as pd
import torch

from .splitter import Splitter, DomainSplitter
from .grouper import Grouper
from .tasks import get_task_config
from .features import PreprocessorConfig
from .metrics import metrics_by_group


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

    @property
    def X_shape(self):
        """Shape of the data matrix for training."""
        return self.data.shape

    @property
    def n_groups(self) -> int:
        """Number of sensitive groups, across all sensitive attributes."""
        return np.prod(self.groups.nunique(axis=0).values)

    def _check_data(self, X: pd.DataFrame, y: pd.Series,
                    g: Union[pd.DataFrame, pd.Series]):
        """Helper function to check data after all preprocessing/splitting."""
        if not pd.api.types.is_numeric_dtype(y):
            print(f"[WARNING] y is of type {y.dtype}; non-numeric types "
                  f"are not accepted by all estimators (e.g. xgb.XGBClassifier")
        return

    def _initialize_data(self):
        """Load the data/labels/groups from a data source."""
        data = self.data_source.get_data().reset_index(drop=True)
        data = self.task_config.feature_list.apply_schema(
            data, passthrough_columns=["Split"])
        data = self.preprocessor_config._dropna(data)
        data = self.grouper.transform(data)
        data = self._generate_splits(data)
        data = self._process_post_split(data)

        X, y, G = self._X_y_G_split(data)
        self._check_data(X, y, G)
        self.data = X
        self.labels = y
        self.groups = G
        del data
        return

    def _X_y_G_split(self, data, default_targets_dtype=int) -> \
            Tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
        """Fetch the (data, labels, groups) arrays from a DataFrame."""
        data_features = [x for x in data.columns
                         if x not in self.grouper.features
                         and x != self.target]
        if not self.grouper.drop:
            # Retain the group variables as features.
            data_features.extend(self.grouper.features)

        if (isinstance(self.splitter, DomainSplitter)
                and self.splitter.drop_domain_split_col):
            # Retain the domain split variable as feature.
            data_features.append(self.splitter.domain_split_varname)

        X = data.loc[:, data_features]
        y = data.loc[:, self.target]
        G = data.loc[:, self.grouper.features]

        if y.dtype == "O":
            y = y.astype(default_targets_dtype)
        return X, y, G

    def _generate_splits(self, data):
        """Call the splitter to generate splits for the dataset."""
        assert self.splits is None, "attempted to overwrite existing splits."

        X_y_g = self._X_y_G_split(data)
        self.splits = self.splitter(*X_y_g)
        data = self._post_split_feature_selection(data)
        return data

    def _post_split_feature_selection(self,
                                      data: pd.DataFrame) -> pd.DataFrame:
        """Select features for post-split processing."""
        if (isinstance(self.splitter, DomainSplitter)
                and self.splitter.drop_domain_split_col):
            # Case: domain split with feature to drop; now that the split has
            # been made, drop the domain split feature.
            data.drop(columns=self.splitter.domain_split_varname, inplace=True)
        if "Split" in data.columns:
            data.drop(columns=["Split"], inplace=True)
        return data

    def _process_post_split(self, data) -> pd.DataFrame:
        """Dataset-specific postprocessing function.

        Conducts any processing required **after** splitting (e.g.
        normalization, drop features needed only for splitting)."""
        passthrough_columns = self.grouper.features + [self.target]

        if (isinstance(self.splitter, DomainSplitter)
                and not self.splitter.drop_domain_split_col):
            passthrough_columns.append(self.splitter.domain_split_varname)

        data = self.preprocessor_config.fit_transform(
            data,
            self.splits["train"],
            passthrough_columns=passthrough_columns)
        return data

    def _check_split(self, split):
        """Check that a split name is valid."""
        assert split in self.splits.keys(), \
            f"split {split} not in {list(self.splits.keys())}"

    def get_pandas(self, split) -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
        """Fetch the (data, labels, groups) for this TabularDataset."""
        self._check_split(split)
        idxs = self.splits[split]
        return (self.data.iloc[idxs],
                self.labels.iloc[idxs],
                self.groups.iloc[idxs])

    def get_dataloader(self, split, batch_size,
                       shuffle=True) -> torch.utils.data.DataLoader:
        """Fetch a dataloader yielding (X, y, G) tuples."""
        self._check_split(split)
        idxs = self.splits[split]
        data = (self.data.iloc[idxs],
                self.labels.iloc[idxs],
                self.groups.iloc[idxs])
        data = tuple(map(lambda x: torch.tensor(x.values).float(), data))
        tds = torch.utils.data.TensorDataset(*data)
        return torch.utils.data.DataLoader(tds, batch_size, shuffle)

    def get_dataset_baseline_metrics(self, split):

        X_tr, y_tr, g = self.get_pandas(split)
        n_by_y = pd.value_counts(y_tr).to_dict()
        y_maj = pd.value_counts(y_tr).idxmax()
        # maps {class_label: p_class_label}
        p_y = pd.value_counts(y_tr, normalize=True).to_dict()

        p_y_by_sens = pd.crosstab(y_tr, [X_tr[c] for c in g],
                                  normalize='columns').to_dict()
        n_y_by_sens = pd.crosstab(y_tr, [X_tr[c] for c in g]).to_dict()
        n_by_sens = pd.crosstab(g.iloc[:, 0], g.iloc[:, 1]).unstack().to_dict()
        return {"y_maj": y_maj,
                "n_by_y": n_by_y,
                "p_y": p_y,
                "p_y_by_sens": p_y_by_sens,
                "n_y_by_sens": n_y_by_sens,
                "n_by_sens": n_by_sens}

    def subgroup_majority_classifier_performance(self, split):
        """Compute overall and worst-group acc of a subgroup-conditional
        majority-class classifier."""
        baseline_metrics = self.get_dataset_baseline_metrics(split)
        sensitive_subgroup_accuracies = []
        sensitive_subgroup_n_correct = []
        for sens, n_y_by_sens in baseline_metrics["n_y_by_sens"].items():
            p_y_by_sens = baseline_metrics["p_y_by_sens"][sens]
            y_max = 1 if n_y_by_sens[1] > n_y_by_sens[0] else 0
            p_y_max = p_y_by_sens[y_max]
            n_y_max = n_y_by_sens[y_max]
            sensitive_subgroup_n_correct.append(n_y_max)
            sensitive_subgroup_accuracies.append(p_y_max)
        n = sum(baseline_metrics["n_by_y"].values())
        overall_acc = np.sum(sensitive_subgroup_n_correct) / n
        return overall_acc, min(sensitive_subgroup_accuracies)

    def evaluate_predictions(self, preds, split):
        _, labels, groups = self.get_pandas(split)
        metrics = metrics_by_group(labels, preds, groups, suffix=split)
        # Add baseline metrics.
        metrics["majority_baseline_" + split] = max(labels.mean(),
                                                    1 - labels.mean())
        sm_overall_acc, sm_wg_acc = self.subgroup_majority_classifier_performance(
            split)
        metrics["subgroup_majority_overall_acc_" + split] = sm_overall_acc
        metrics["subgroup_majority_worstgroup_acc_" + split] = sm_wg_acc
        return metrics

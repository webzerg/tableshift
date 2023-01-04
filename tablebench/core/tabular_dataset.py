from abc import ABC
from dataclasses import dataclass
import glob
import json
import math
import os
import pickle
from typing import Optional, Tuple, Union, List, Dict, Any

import numpy as np
import pandas as pd
from pandas import DataFrame, Series
import ray.data
import torch
from torch.utils.data import DataLoader
from torch.utils.data.dataloader import default_collate

from .splitter import Splitter, DomainSplitter
from .grouper import Grouper
from .tasks import get_task_config
from .features import PreprocessorConfig
from .metrics import metrics_by_group
from .utils import make_uid
from tablebench.third_party.domainbed import InfiniteDataLoader


def _make_dataloader_from_dataframes(
        data, batch_size: int, shuffle: bool,
        device: str, infinite=False) -> DataLoader:
    """Construct a DataLoader from a DataFrame."""
    device = torch.device(device)
    data = tuple(map(lambda x: torch.tensor(x.values).float(), data))
    tds = torch.utils.data.TensorDataset(*data)
    _collate_fn = lambda x: tuple(t.to(device) for t in default_collate(x))
    if infinite:
        loader = InfiniteDataLoader(dataset=tds, batch_size=batch_size,
                                    weights=None,
                                    shuffle=shuffle,
                                    collate_fn=_collate_fn)
    else:
        loader = DataLoader(
            dataset=tds, batch_size=batch_size,
            shuffle=shuffle,
            # num_workers=num_workers,
            collate_fn=_collate_fn)
    return loader


@dataclass
class TabularDatasetConfig:
    cache_dir: str = "tmp"
    download: bool = True
    random_seed: int = 948324


class TabularDataset(ABC):
    def __init__(self, name: str, config: TabularDatasetConfig,
                 splitter: Splitter,
                 preprocessor_config: PreprocessorConfig,
                 grouper: Optional[Grouper], initialize_data=True,
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
        self._df: pd.DataFrame = None  # holds all the data

        self.splits = None  # dict mapping {split_name: list of idxs in split}
        if initialize_data:
            self._initialize_data()

    @property
    def features(self):
        return self.task_config.feature_list.names

    @property
    def predictors(self) -> List[str]:
        """The list of feature names in the FeatureList.

        Note that these do *not* necessarily correspond to the names in X,
        the data provided after preprocessing."""
        return self.task_config.feature_list.predictors

    @property
    def X_shape(self):
        """Shape of the data matrix for training."""
        return [None, len(self.feature_names)]

    @property
    def n_domains(self) -> int:
        """Number of domains, across all sensitive attributes."""
        if self.domain_label_colname is None:
            return 0
        else:
            return self._df[self.domain_label_colname].nunique()

    @property
    def eval_split_names(self) -> Tuple:
        """Fetch the names of the eval splits."""
        return tuple([x for x in self.splits.keys() if "train" not in x])

    def _check_data(self):
        """Helper function to check data after all preprocessing/splitting."""
        if not pd.api.types.is_numeric_dtype(self._df[self.target]):
            print(f"[WARNING] y is of type {self._df[self.target].dtype}; "
                  f"non-numeric types are not accepted by all estimators ("
                  f"e.g. xgb.XGBClassifier")
        if self.domain_label_colname:
            assert self.domain_label_colname not in self._df[
                self.feature_names].columns

        if self.grouper.drop:
            for c in self.grouper.features: assert c not in self._df[
                self.feature_names].columns
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
        self._df = data

        self._init_feature_names(data)
        self._check_data()

        return

    def _init_feature_names(self, data):
        """Set the (data, labels, groups, domain_labels) feature names."""
        target = self.task_config.feature_list.target
        data_features = set([x for x in data.columns
                             if x not in self.grouper.features
                             and x != target])
        if not self.grouper.drop:
            # Retain the group variables as features.
            for x in self.grouper.features: data_features.add(x)

        if isinstance(self.splitter, DomainSplitter):
            domain_split_varname = self.splitter.domain_split_varname

            if self.splitter.drop_domain_split_col and \
                    (domain_split_varname in data_features):
                # Retain the domain split variable as feature in X.
                data_features.remove(domain_split_varname)
        else:
            # Case: domain split is not used; no domain labels exist.
            domain_split_varname = None

        self.feature_names = list(data_features)
        self.target = target
        self.group_feature_names = self.grouper.features
        self.domain_label_colname = domain_split_varname

        return

    def _generate_splits(self, data):
        """Call the splitter to generate splits for the dataset."""
        assert self.splits is None, "attempted to overwrite existing splits."

        self._init_feature_names(data)
        self.splits = self.splitter(
            data=data[self.feature_names],
            labels=data[self.target],
            groups=data[self.group_feature_names],
            domain_labels=data[self.domain_label_colname] \
                if self.domain_label_colname else None)
        if "Split" in data.columns:
            data.drop(columns=["Split"], inplace=True)
        return data

    def _process_post_split(self, data,
                            default_targets_dtype=int) -> pd.DataFrame:
        """Dataset-specific postprocessing function.

        Conducts any processing required **after** splitting (e.g.
        normalization, drop features needed only for splitting)."""
        passthrough_columns = self.grouper.features + [self.target]

        data = self.preprocessor_config.fit_transform(
            data,
            self.splits["train"],
            domain_label_colname=self.domain_label_colname,
            passthrough_columns=passthrough_columns)
        if data[self.task_config.feature_list.target].dtype == "O":
            data[self.task_config.feature_list.target] = data[
                self.task_config.feature_list.target].astype(
                default_targets_dtype)
        return data

    def _check_split(self, split):
        """Check that a split name is valid."""
        assert split in self.splits.keys(), \
            f"split {split} not in {list(self.splits.keys())}"

    def _get_split_idxs(self, split):
        self._check_split(split)
        idxs = self.splits[split]
        return idxs

    def _get_split_df(self, split):
        idxs = self._get_split_idxs(split)
        return self._df.iloc[idxs]

    def _get_split_xygd(self, split) -> Tuple[
        DataFrame, Series, DataFrame, Optional[Series]]:
        df = self._get_split_df(split)
        X = df[self.feature_names]
        y = df[self.target]
        G = df[self.group_feature_names]
        d = df[self.domain_label_colname] \
            if self.domain_label_colname is not None else None
        return X, y, G, d

    def get_pandas(self, split) -> Tuple[
        DataFrame, Series, DataFrame, Optional[Series]]:
        """Fetch the (data, labels, groups, domains) for this TabularDataset."""

        # TODO(jpgard): consider naming these outputs, or creating
        #  a DataClass object to "hold" them. This will allow for easy access of
        #  e.g. numeric vs. categorical features, where this is needed.
        return self._get_split_xygd(split)

    def get_domain_dataloaders(self, split, batch_size=2048, device='cpu',
                               shuffle=True, infinite=True) -> Dict[
        Any, DataLoader]:
        """Fetch a dict of {domain_id:DataLoader}."""
        loaders = {}
        data = self._get_split_xygd(split)
        assert self.n_domains, "sanity check for a domain-split dataset"

        for domain in data[self.domain_label_colname].unique():
            split_domain_data = data[data[self.domain_label_colname] == domain]
            split_loader = _make_dataloader_from_dataframes(
                split_domain_data, batch_size, shuffle, device,
                infinite=infinite)
            loaders[domain] = split_loader
        return loaders

    def get_dataloader(self, split, batch_size=2048, device='cpu',
                       shuffle=True, infinite=False) -> DataLoader:
        """Fetch a dataloader yielding (X, y, G, d) tuples."""
        data = self._get_split_xygd(split)
        if not self.domain_label_colname:
            # Drop the empty domain labels.
            data = data[:-1]
        return _make_dataloader_from_dataframes(data, batch_size, shuffle,
                                                device, infinite=infinite)

    def get_dataset_baseline_metrics(self, split):

        X_tr, y_tr, g, _ = self.get_pandas(split)
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
        _, labels, groups, _ = self.get_pandas(split)
        metrics = metrics_by_group(labels, preds, groups, suffix=split)
        # Add baseline metrics.
        metrics["majority_baseline_" + split] = max(labels.mean(),
                                                    1 - labels.mean())
        sm_overall_acc, sm_wg_acc = self.subgroup_majority_classifier_performance(
            split)
        metrics["subgroup_majority_overall_acc_" + split] = sm_overall_acc
        metrics["subgroup_majority_worstgroup_acc_" + split] = sm_wg_acc
        return metrics

    def is_cached(self) -> bool:
        uid = make_uid(self.name, self.splitter)
        base_dir = os.path.join(self.config.cache_dir, uid)
        if os.path.exists(os.path.join(base_dir, "info.json")):
            return True
        else:
            return False

    def to_sharded(self, rows_per_shard=8192):
        uid = make_uid(self.name, self.splitter)

        base_dir = os.path.join(self.config.cache_dir, uid)
        for split in self.splits:
            outdir = os.path.join(base_dir, split)
            print(f"[INFO] caching task {uid} to {outdir}")
            if not os.path.exists(outdir):
                os.makedirs(outdir)
            df = self._get_split_df(split)

            num_shards = math.ceil(len(df) / rows_per_shard)
            for i in range(num_shards):
                fp = os.path.join(outdir, f"{split}_{i:05d}.csv")
                df.iloc[i * rows_per_shard:(i + 1) * rows_per_shard].to_csv(fp,
                                                                            index=False)

        # write metadata
        schema = self._df.dtypes.to_dict()
        with open(os.path.join(base_dir, "schema.pickle"), "wb") as f:
            pickle.dump(schema, f)

        ds_info = {
            'target': self.target,
            'domain_label_colname': self.domain_label_colname,
            'group_feature_names': self.group_feature_names,
            'feature_names': self.feature_names,
            'X_shape': self.X_shape,
            'splits': list(self.splits.keys())
        }
        with open(os.path.join(base_dir, "info.json"), "w") as f:
            f.write(json.dumps(ds_info))


# TODO(jpgard): CachedDataset and TabularDataset should inherit from a shared
#  parent Dataset class.
class CachedDataset:
    def __init__(self, cache_dir: str, name: str, uid: str):
        self.cache_dir = cache_dir
        self.uid = uid
        self.name = name
        self.target = None
        self.domain_label_colname = None
        self.group_feature_names = None
        self.feature_names = None
        self.X_shape = None
        self.splits: List = None

        self._load_info_from_cache()

    @property
    def base_dir(self):
        return os.path.join(self.cache_dir, self.uid)

    def is_cached(self) -> bool:
        base_dir = os.path.join(self.cache_dir, self.uid)
        if os.path.exists(os.path.join(base_dir, "info.json")):
            return True
        else:
            return False

    def _load_info_from_cache(self):
        print(f"[INFO] loading from {self.base_dir}")
        with open(os.path.join(self.base_dir, "info.json"), "r") as f:
            ds_info = json.loads(f.read())

        for k, v in ds_info.items():
            setattr(self, k, v)

        with open(os.path.join(self.base_dir, "schema.pickle"), "rb") as f:
            schema = pickle.load(f)

    def get_ray(self, split, num_partitions=64):
        dir = os.path.join(self.base_dir, split)
        fileglob = os.path.join(dir, "*.csv")
        files = glob.glob(fileglob)
        assert len(files), f"no files detected for split {split} " \
                           f"matching {fileglob}"
        return ray.data \
            .read_csv(files,
                      meta_provider=ray.data.datasource.FastFileMetadataProvider()) \
            .repartition(num_partitions)

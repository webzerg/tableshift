from abc import ABC, abstractmethod
from dataclasses import dataclass
import glob
import json
import logging
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

from .splitter import Splitter, DomainSplitter
from .grouper import Grouper
from .tasks import get_task_config
from .features import Preprocessor, PreprocessorConfig
from .metrics import metrics_by_group
from .utils import make_uid, convert_64bit_numeric_cols
from tableshift.third_party.domainbed import InfiniteDataLoader


def _make_dataloader_from_dataframes(
        data, batch_size: int, shuffle: bool,
        infinite=False) -> DataLoader:
    """Construct a (shuffled) DataLoader from a DataFrame."""
    data = tuple(map(lambda x: torch.tensor(x.values).float(), data))
    tds = torch.utils.data.TensorDataset(*data)
    if infinite:
        loader = InfiniteDataLoader(dataset=tds, batch_size=batch_size)
    else:
        loader = DataLoader(
            dataset=tds, batch_size=batch_size,
            shuffle=shuffle)
    return loader


@dataclass
class TabularDatasetConfig:
    cache_dir: str = "tableshift_cache"
    download: bool = True
    random_seed: int = 948324


@dataclass
class Dataset(ABC):
    """Absract class to represent a dataset."""
    name: str
    preprocessor_config: PreprocessorConfig
    splitter: Splitter = None
    splits = None  # dict mapping {split_name: list of idxs in split}

    feature_names: Union[List[str], None] = None
    group_feature_names: Union[List[str], None] = None
    target: str = None

    domain_label_colname: Union[str, None] = None

    # If true, do not do per-domain evals
    skip_per_domain_eval: bool = False

    @property
    def uid(self) -> str:
        return make_uid(self.name, self.splitter)

    @property
    def is_domain_split(self) -> bool:
        """Return True if this dataset uses a DomainSplitter, else False."""
        return self.domain_label_colname is not None

    @property
    def eval_split_names(self) -> Tuple:
        """Fetch the names of the eval splits."""
        if self.skip_per_domain_eval:
            return tuple([x for x in self.splits if
                          x in ("test", "id_test", "ood_test")])

        else:
            return tuple([x for x in self.splits if "train" not in x])

    @property
    def domain_split_varname(self):
        if not self.is_domain_split:
            return None

        elif isinstance(self.splitter, DomainSplitter):
            return self.splitter.domain_split_varname
        else:
            return self.domain_label_colname

    @property
    @abstractmethod
    def n_domains(self) -> int:
        raise

    @property
    @abstractmethod
    def base_dir(self) -> str:
        "Return the location of the directory {cache_dir}/{uid}."
        raise

    @abstractmethod
    def _is_valid_split(self, split) -> bool:
        raise

    def _check_split(self, split):
        """Check that a split name is valid."""
        assert self._is_valid_split(split), \
            f"split {split} not in {list(self.splits.keys())}"

    @abstractmethod
    def _get_split_df(self, split: str) -> pd.DataFrame:
        raise

    def _get_split_xygd(self, split) -> Tuple[
        DataFrame, Series, DataFrame, Optional[Series]]:
        for name in ("feature_names", "target", "group_feature_names"):
            assert getattr(self, name) is not None, f"{name} is None."
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
        return self._get_split_xygd(split)

    def get_dataloader(self, split, batch_size=2048,
                       shuffle=True, infinite=False) -> DataLoader:
        """Fetch a dataloader yielding (X, y, G, d) tuples."""
        data = self._get_split_xygd(split)
        if not self.domain_label_colname:
            # Drop the empty domain labels.
            data = data[:-1]
        return _make_dataloader_from_dataframes(data, batch_size, shuffle,
                                                infinite=infinite)

    def get_cache_dir(self, split: str, domain: Optional[Any] = None):
        if domain is None:
            return os.path.join(self.base_dir, split)
        else:
            return os.path.join(self.base_dir, split, str(domain))


class TabularDataset(Dataset):
    def __init__(self, name: str, config: TabularDatasetConfig,
                 splitter: Splitter,
                 preprocessor_config: PreprocessorConfig,
                 grouper: Optional[Grouper], initialize_data=True,
                 **kwargs):
        super().__init__(name=name,
                         preprocessor_config=preprocessor_config,
                         splitter=splitter)
        self.config = config
        self.grouper = grouper

        # Dataset-specific info: features, data source, preprocessing.

        self.task_config = get_task_config(self.name)
        self.data_source = self.task_config.data_source_cls(
            cache_dir=self.config.cache_dir,
            download=self.config.download,
            **kwargs)

        self.preprocessor = Preprocessor(
            config=self.preprocessor_config,
            feature_list=self.task_config.feature_list)

        # Placeholders for data/labels/groups and split indices.
        self._df: Union[pd.DataFrame, None] = None  # holds all the data

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
    def grouper_features(self):
        if self.grouper is not None:
            return self.grouper.features
        else:
            return []

    @property
    def n_train(self) -> int:
        """Fetch the number of training observations."""
        return len(self.splits["train"])

    @property
    def n_domains(self) -> int:
        """Number of domains, across all sensitive attributes and splits."""
        if self.domain_label_colname is None:
            return 0
        else:
            return self._df[self.domain_label_colname].nunique()

    def get_domains(self, split) -> Union[List[str], None]:
        """Fetch a list of the domains."""
        if self.is_domain_split and self._is_valid_split(split):
            split_df = self._get_split_df(split)
            return split_df[self.domain_label_colname].unique()
        else:
            return None

    def _check_data(self):
        """Helper function to check data after all preprocessing/splitting."""
        if not pd.api.types.is_numeric_dtype(self._df[self.target]):
            logging.warning(
                f"y is of type {self._df[self.target].dtype}; "
                f"non-numeric types are not accepted by all estimators ("
                f"e.g. xgb.XGBClassifier")
        if self.domain_label_colname:
            assert self.domain_label_colname not in self._df[
                self.feature_names].columns

        if self.grouper and self.grouper.drop:
            for c in self.grouper_features: assert c not in self._df[
                self.feature_names].columns
        return

    @staticmethod
    def _check_data_source(df: pd.DataFrame):
        """Check the data returned by DataSource.get_data()."""
        df = convert_64bit_numeric_cols(df)
        df.reset_index(drop=True, inplace=True)
        return df

    def _initialize_data(self):
        """Load the data/labels/groups from a data source."""
        data = self.data_source.get_data()
        data = self._check_data_source(data)
        data = self.task_config.feature_list.apply_schema(
            data, passthrough_columns=["Split"])
        data = self.preprocessor._dropna(data)
        data = self._apply_grouper(data)
        data = self._generate_splits(data)
        data = self._process_post_split(data)
        self._df = data

        self._init_feature_names(data)
        self._check_data()

        return

    def _apply_grouper(self, data: pd.DataFrame):
        """Apply the grouper, if one is used."""
        if self.grouper is not None:
            return self.grouper.transform(data)
        else:
            return data

    def _init_feature_names(self, data):
        """Set the (data, labels, groups, domain_labels) feature names."""
        target = self.task_config.feature_list.target
        data_features = set([x for x in data.columns
                             if x not in self.grouper_features
                             and x != target])
        if self.grouper and (not self.grouper.drop):
            # Retain the group variables as features.
            for x in self.grouper_features: data_features.add(x)

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
        self.group_feature_names = self.grouper_features
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
        passthrough_columns = self.grouper_features + [self.target]

        data = self.preprocessor.fit_transform(
            data,
            self.splits["train"],
            domain_label_colname=self.domain_label_colname,
            passthrough_columns=passthrough_columns)
        if data[self.task_config.feature_list.target].dtype == "O":
            data[self.task_config.feature_list.target] = data[
                self.task_config.feature_list.target].astype(
                default_targets_dtype)
        return data

    def _is_valid_split(self, split) -> bool:
        return split in self.splits.keys()

    def _get_split_idxs(self, split):
        self._check_split(split)
        idxs = self.splits[split]
        return idxs

    def _get_split_df(self, split) -> pd.DataFrame:
        idxs = self._get_split_idxs(split)
        return self._df.iloc[idxs]

    def get_domain_dataloaders(
            self, split, batch_size=2048,
            shuffle=True, infinite=True) -> Dict[Any, DataLoader]:
        """Fetch a dict of {domain_id:DataLoader}."""
        loaders = {}
        split_data = self._get_split_xygd(split)
        assert self.n_domains, "sanity check for a domain-split dataset"

        logging.debug("Domain value counts:\n{}".format(
            split_data[3].value_counts()))

        for domain in sorted(split_data[3].unique()):
            # Boolean vector where True indicates observations in the domain.
            idxs = split_data[3] == domain
            assert idxs.sum() >= batch_size, \
                "sanity check at least one full batch per domain."

            split_domain_data = [df[idxs] for df in split_data]
            split_loader = _make_dataloader_from_dataframes(
                split_domain_data, batch_size, shuffle, infinite=infinite)
            loaders[domain] = split_loader
        return loaders

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
        base_dir = os.path.join(self.config.cache_dir, self.uid)
        if os.path.exists(os.path.join(base_dir, "info.json")):
            return True
        else:
            return False

    @property
    def base_dir(self) -> str:
        return os.path.join(self.config.cache_dir, self.uid)

    def to_sharded(self, rows_per_shard=4096,
                   domains_to_subdirectories: bool = True):

        base_dir = self.base_dir

        def initialize_dir(dirname):
            if not os.path.exists(dirname):
                os.makedirs(dirname)

        for split in self.splits:

            logging.info(f"caching task split {split} to {self.base_dir}")

            df = self._get_split_df(split)

            def write_shards(df, dirname):
                num_shards = math.ceil(len(df) / rows_per_shard)
                for i in range(num_shards):
                    fp = os.path.join(dirname, f"{split}_{i:05d}.csv")
                    logging.debug('writing file to %s' % fp)
                    df.iloc[i * rows_per_shard:(i + 1) * rows_per_shard] \
                        .to_csv(fp, index=False)

            if self.domain_label_colname and domains_to_subdirectories:
                # Write to {split}/{domain_value}/{shard_filename.csv}
                for domain in sorted(df[self.domain_label_colname].unique()):
                    df_ = df[df[self.domain_label_colname] == domain]
                    domain_dir = self.get_cache_dir(split, domain)
                    initialize_dir(domain_dir)
                    write_shards(df_, domain_dir)
            elif self.domain_label_colname:
                # Write to {split}/all_{split}_subdomains/{shard_filename.csv}
                shared_domain_dir = self.get_cache_dir(
                    split, f"all_{split}_subdomains")
                initialize_dir(shared_domain_dir)
                write_shards(df, shared_domain_dir)
            else:
                # Write to {split}/{shard_filename.csv}
                outdir = self.get_cache_dir(split)
                initialize_dir(outdir)
                write_shards(df, outdir)

        # write metadata
        schema = self._df.dtypes.to_dict()
        with open(os.path.join(base_dir, "schema.pickle"), "wb") as f:
            pickle.dump(schema, f)

        ds_info = {
            'target': self.target,
            'domain_label_colname': self.domain_label_colname,
            'domain_label_values': self._df[
                self.domain_label_colname].unique().tolist() \
                if self.domain_label_colname else None,
            'group_feature_names': self.group_feature_names,
            'feature_names': self.feature_names,
            'X_shape': self.X_shape,
            'splits': list(self.splits.keys()),
            **{f'n_{s}': len(self.splits[s]) for s in self.splits},
        }
        with open(os.path.join(base_dir, "info.json"), "w") as f:
            f.write(json.dumps(ds_info))


class CachedDataset(Dataset):
    def __init__(self, cache_dir: str, **kwargs):
        super().__init__(**kwargs)
        self.cache_dir = cache_dir

        self.domain_label_values = None
        self.group_feature_names = None
        self.X_shape = None

        self.schema = None

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
        """Load the dataset metadata from cache (data is lazily loaded)."""
        logging.info(f"loading from {self.base_dir}")
        with open(os.path.join(self.base_dir, "info.json"), "r") as f:
            ds_info = json.loads(f.read())

        for k, v in ds_info.items():
            setattr(self, k, v)

        with open(os.path.join(self.base_dir, "schema.pickle"), "rb") as f:
            self.schema = pickle.load(f)

    def get_domains(self, split) -> Union[List[str], None]:
        """Fetch a list of the cached domains."""
        dir = os.path.join(self.base_dir, split)
        if not os.path.exists(dir):
            return None
        else:
            domains = os.listdir(dir)
            return sorted(domains)

    @property
    def n_domains(self) -> int:
        """Number of domains, across all sensitive attributes and splits."""
        if self.domain_label_colname is None:
            return 0
        else:
            domains_per_split = [self.get_domains(s) for s in self.splits]
            domains = list(set(d for ds in domains_per_split for d in ds))
            return len(domains)

    def _get_split_files(self, split: str, domain: Optional[str] = None):
        cache_dir = self.get_cache_dir(split, domain)
        fileglob = os.path.join(cache_dir, "*.csv")
        files = glob.glob(fileglob)

        assert len(files), f"no files detected for split {split} " \
                           f"matching {fileglob}"
        return files

    def _is_valid_split(self, split) -> bool:
        return split in os.listdir(self.base_dir)

    def _get_split_df(self, split) -> pd.DataFrame:
        self._check_split(split)
        files = self._get_split_files(split)
        dfs = []
        for f in files:
            dfs.append(pd.read_csv(f))
        df = pd.concat(dfs)

        return df

    def get_ray(self, split, domain=None, num_partitions_per_file=16):
        files = self._get_split_files(split, domain)
        num_partitions = len(files) * num_partitions_per_file
        return ray.data \
            .read_csv(
            files,
            meta_provider=ray.data.datasource.FastFileMetadataProvider()) \
            .repartition(num_partitions)

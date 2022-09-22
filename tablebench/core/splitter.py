from dataclasses import dataclass
from typing import Sequence, Mapping, Any, List, Optional

import pandas as pd

import numpy as np
from sklearn.utils import indexable
from sklearn.utils.validation import _num_samples
from sklearn.model_selection._split import _validate_shuffle_split, \
    StratifiedShuffleSplit, ShuffleSplit


def train_test_split(
        *arrays,
        test_size=None,
        train_size=None,
        random_state=None,
        shuffle=True,
        stratify=None):
    """Fork of sklearn.model_selection.train_test_split that returns indices."""
    n_arrays = len(arrays)
    if n_arrays == 0:
        raise ValueError("At least one array required as input")

    arrays = indexable(*arrays)

    n_samples = _num_samples(arrays[0])
    n_train, n_test = _validate_shuffle_split(
        n_samples, test_size, train_size, default_test_size=0.25
    )

    if shuffle is False:
        if stratify is not None:
            raise ValueError(
                "Stratified train/test split is not implemented for shuffle=False"
            )

        train = np.arange(n_train)
        test = np.arange(n_train, n_train + n_test)

    else:
        if stratify is not None:
            CVClass = StratifiedShuffleSplit
        else:
            CVClass = ShuffleSplit

        cv = CVClass(test_size=n_test, train_size=n_train,
                     random_state=random_state)

        train, test = next(cv.split(X=arrays[0], y=stratify))
    return train, test


def concat_columns(data: pd.DataFrame) -> pd.DataFrame:
    """Helper function to concatenate values over a set of columns.

    This is useful, for example, as a preprocessing step for performing
    stratified sampling over labels + sensitive attributes."""
    return data.agg(lambda x: ''.join(x.values.astype(str)), axis=1).T


@dataclass
class Splitter:
    """Splitter for non-domain splits."""
    val_size: float
    random_state: int

    def __call__(self, data: pd.DataFrame, labels: pd.Series,
                 groups: pd.DataFrame = None, *args, **kwargs) -> Mapping[
        str, List[int]]:
        """Split a dataset.

        Returns a dict mapping split names to indices of the data points
        in that split."""
        raise


class FixedSplitter(Splitter):
    """A splitter for using fixed splits.

    This occurs, for example, when a dataset has a fixed train-test
    split (such as the Adult dataset).

    The FixedSplitter assumes there is a column in the dataset, "Split",
    which contains the values "train", "test".

    Note that for the fixed splitter, val_size indicates what fraction
    **of the training data** should be used for the validation set
    (since we cannot control the fraction of the overall data dedicated
    to validation, due to the prespecified train/test split).
    """

    def __call__(self, data: pd.DataFrame, labels: pd.Series,
                 groups: pd.DataFrame = None, *args, **kwargs) -> Mapping[
        str, List[int]]:
        test_idxs = np.nonzero((data["Split"] == "test").values)[0]
        train_val_idxs = np.nonzero((data["Split"] == "train").values)[0]

        stratify = labels
        if groups is not None:
            stratify = pd.concat((stratify, groups), axis=1)

        train_idxs, val_idxs = train_test_split(
            data.iloc[train_val_idxs],
            train_size=(1 - self.val_size),
            random_state=self.random_state,
            stratify=stratify.iloc[train_val_idxs])

        del train_val_idxs
        return {"train": train_idxs, "validation": val_idxs, "test": test_idxs}


@dataclass
class RandomSplitter(Splitter):
    test_size: float

    @property
    def train_size(self):
        return 1. - (self.val_size + self.test_size)

    def __call__(self, data: pd.DataFrame, labels: pd.Series,
                 groups: pd.DataFrame = None, *args, **kwargs
                 ) -> Mapping[str, List[int]]:
        # Build a stratification DataFrame using the labels and (optionally)
        # groups. This ensures that splitting happens approximately
        # uniformly over labels and features.

        stratify = labels
        if groups is not None:
            stratify = pd.concat((stratify, groups), axis=1)
        stratify = concat_columns(stratify)

        train_val_idxs, test_idxs = train_test_split(
            data,
            test_size=self.test_size,
            random_state=self.random_state,
            stratify=stratify)
        train_idxs, val_idxs = train_test_split(
            data.iloc[train_val_idxs],
            train_size=self.train_size / (self.train_size + self.val_size),
            random_state=self.random_state,
            stratify=stratify.iloc[train_val_idxs])
        del train_val_idxs
        return {"train": train_idxs, "validation": val_idxs, "test": test_idxs}


@dataclass
class DomainSplitter(RandomSplitter):
    """Splitter for domain splits."""
    domain_split_varname: str
    domain_split_ood_values: Sequence[str]

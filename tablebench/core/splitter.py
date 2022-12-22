from abc import abstractmethod
from dataclasses import dataclass
from typing import Sequence, Mapping, Any, List, Optional, Tuple

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


def _stratify(labels, groups):
    """Make a stratification vector by concatenating label and (optional) groups.

    This ensures that splitting happens approximately uniformly over
    labels and sensitive subgroups.
    """
    strata = labels
    if groups is not None:
        strata = pd.concat((strata, groups), axis=1)
    strata = concat_columns(strata)
    return strata


@dataclass
class Splitter:
    """Splitter for non-domain splits."""
    val_size: float
    random_state: int

    @abstractmethod
    def __call__(self, data: pd.DataFrame, labels: pd.Series,
                 groups: pd.DataFrame = None, *args, **kwargs) -> Mapping[
        str, List[int]]:
        """Split a dataset.

        Returns a dictionary mapping split names to indices of the data points
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
        assert "Split" in data.columns, "data is missing 'Split' column."
        test_idxs = np.nonzero((data["Split"] == "test").values)[0]
        train_val_idxs = np.nonzero((data["Split"] == "train").values)[0]

        stratify = _stratify(labels, groups)

        train_idxs, val_idxs = train_test_split(
            data.iloc[train_val_idxs],
            train_size=(1 - self.val_size),
            random_state=self.random_state,
            stratify=stratify.iloc[train_val_idxs])

        del train_val_idxs
        return {"train": train_idxs, "validation": val_idxs, "test": test_idxs}


def _check_input_indices(data: pd.DataFrame):
    """Helper function to validate input indices.

    If a DataFrame is not indexed from (0, n), which happens e.g. when the
    DataFrame has been filtered without resetting the index, it can cause
    major downstream issues with splitting. This is because splitting will
    assume that all values (0,...n) are represented in the index.
    """
    idxs = np.array(sorted(data.index.tolist()))
    expected = np.arange(len(data))
    assert np.all(idxs == expected), "DataFrame is indexed non-sequentially;" \
                                     "try passing the dataframe after "
    return


@dataclass
class RandomSplitter(Splitter):
    test_size: float

    @property
    def train_size(self):
        return 1. - (self.val_size + self.test_size)

    def __call__(self, data: pd.DataFrame, labels: pd.Series,
                 groups: pd.DataFrame = None, *args, **kwargs
                 ) -> Mapping[str, List[int]]:
        _check_input_indices(data)
        stratify = _stratify(labels, groups)

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
class DomainSplitter(Splitter):
    """Splitter for domain splits.

    All observations with domain_split_varname values in domain_split_ood_values
    are placed in the target (test) set; the remaining observations are split
    between the train, validation, and eval set.
    """
    id_test_size: float  # The in-domain test set.
    domain_split_varname: str
    domain_split_ood_values: Sequence[Any]
    domain_split_id_values: Optional[Sequence[Any]] = None
    drop_domain_split_col: bool = True  # If True, drop column after splitting.
    ood_val_size: float = 0  # Fraction of OOD data to use for OOD validation set.

    def __call__(self, data: pd.DataFrame, labels: pd.Series,
                 groups: pd.DataFrame = None, *args, **kwargs) -> Mapping[
        str, List[int]]:
        assert "domain_labels" in kwargs, "domain labels are required."
        domain_vals = kwargs.pop("domain_labels")
        assert isinstance(domain_vals, pd.Series)

        assert isinstance(self.domain_split_ood_values, tuple) \
               or isinstance(self.domain_split_ood_values, list), \
            "domain_split_ood_values must be an iterable type; got type {}".format(
                type(self.domain_split_ood_values))

        def _idx_where_in(x: pd.Series, vals: Sequence[Any],
                          negate=False) -> np.ndarray:
            """Return a vector of the numeric indices i where X[i] in vals.

            If negate, return the vector of indices i where X[i] not in vals."""
            assert isinstance(vals, list) or isinstance(vals, tuple)
            idxs_bool = x.isin(vals)
            idxs_in = np.nonzero(idxs_bool.values)[0]
            if negate:
                return ~idxs_in
            else:
                return idxs_in

        ood_vals = self.domain_split_ood_values

        # Fetch the out-of-domain indices.
        ood_idxs = _idx_where_in(domain_vals, ood_vals)

        # Fetch the in-domain indices; these are either the explicitly-specified
        # in-domain values, or any values not in the OOD values.

        if self.domain_split_id_values is not None:
            # Check that there is no overlap between train/test domains.
            assert not len(
                set(self.domain_split_id_values).intersection(
                    set(ood_vals)))

            id_idxs = _idx_where_in(domain_vals, self.domain_split_id_values)
            if not len(id_idxs):
                raise ValueError(
                    f"No ID observations with {self.domain_split_varname} "
                    f"values {self.domain_split_id_values}; are the values of "
                    f"same type as the column type of {domain_vals.dtype}?")
        else:
            id_idxs = _idx_where_in(domain_vals, ood_vals,
                                    negate=True)
            if not len(id_idxs):
                raise ValueError(
                    f"No ID observations with {self.domain_split_varname} "
                    f"values not in {ood_vals}.")

        if not len(ood_idxs):
            vals = domain_vals.unique()
            raise ValueError(
                f"No OOD observations with {self.domain_split_varname} values "
                f"{ood_vals}; are the values of same type"
                f" as the column type of {domain_vals.dtype}? Examples of "
                f"values in {self.domain_split_varname}: {vals[:10]}")

        stratify = _stratify(labels, groups)

        train_idxs, id_valid_eval_idxs = train_test_split(
            data.iloc[id_idxs],
            test_size=(self.val_size + self.id_test_size),
            random_state=self.random_state,
            stratify=stratify.iloc[id_idxs])

        valid_idxs, id_test_idxs = train_test_split(
            data.loc[id_valid_eval_idxs],
            test_size=self.id_test_size / (self.val_size + self.id_test_size),
            random_state=self.random_state,
            stratify=stratify.iloc[id_valid_eval_idxs])

        outputs = {"train": train_idxs, "validation": valid_idxs,
                   "id_test": id_test_idxs}

        if self.ood_val_size:
            ood_test_idxs, ood_valid_idxs = train_test_split(
                data.loc[ood_idxs],
                test_size=self.ood_val_size,
                random_state=self.random_state,
                stratify=stratify.iloc[ood_idxs])
            outputs["ood_test"] = ood_test_idxs
            outputs["ood_validation"] = ood_valid_idxs

        else:
            outputs["ood_test"] = ood_idxs

        return outputs

from dataclasses import dataclass, field
from functools import partial
import logging
from typing import List, Any, Sequence, Optional, Mapping, Tuple, Union, Dict

import numpy as np
import pandas as pd
from pandas.api.types import CategoricalDtype as cat_dtype
from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.preprocessing import OneHotEncoder, StandardScaler, MinMaxScaler, \
    LabelEncoder, FunctionTransformer
from tqdm import tqdm
from tableshift.core.discretization import KBinsDiscretizer
from tableshift.core.utils import sub_illegal_chars


def _contains_missing_values(df: pd.DataFrame) -> bool:
    return np.any(pd.isnull(df).values)


def safe_cast(x: pd.Series, dtype):
    logging.debug(f"casting feature {x.name} from dtype "
                  f"{x.dtype.name} to dtype {dtype.__name__}")
    if dtype == cat_dtype:
        return x.apply(str).astype("category")
    else:
        try:
            return x.astype(dtype)
        except pd.errors.IntCastingNaNError as e:
            if 'int' in dtype.__name__.lower():
                # Case: integer with nan values; cast to float instead. Integers
                # are not nullable in Pandas (but "Int64" type is).
                logging.warning(
                    f"cannot cast feature {x.name} to "
                    f"dtype {dtype.__name__} due to missing values; "
                    f"attempting cast to float instead. Recommend changing"
                    f"the feature spec for this feature to type float.")
                return x.astype(float)


def _is_categorical(x: pd.Series):
    return isinstance(x.dtype, cat_dtype) or isinstance(x, pd.Categorical)


def column_is_of_type(x: pd.Series, dtype) -> bool:
    """Helper function to check whether column has specified dtype."""
    if hasattr(dtype, "name"):
        # Case: target dtype is of categorical dtype.
        return (hasattr(x.dtype, "name")) and (x.dtype.name == dtype.name)
    elif _is_categorical(x):
        # Case: input data is of categorical dtype.
        return dtype == cat_dtype
    else:
        # Check if x is a subdtype of the more general type specified in
        # dtype; this will not perform casting of identical subtypes (i.e.
        # does not cast int64 to int).
        try:
            return np.issubdtype(x.dtype, dtype)
        except Exception as e:
            logging.error(e)
            import ipdb;
            ipdb.set_trace()


@dataclass(frozen=True)
class Feature:
    name: str
    kind: Any  # a data type to which the feature should be castable.
    description: str = None
    is_target: bool = False
    # Values, besides np.nan, to count as null/missing. These should be
    # values in the original feature encoding (that is, the values that would
    # occur for this column in the output of the preprocess_fn, not after
    # casting to `kind`), because values after casting may be unpredictable.
    na_values: Tuple = field(default_factory=tuple)
    # Mapping of the set of values in the data to a more descriptive set of
    # values. Used e.g. for categorical features that are coded with numeric
    # values that map to known/named categories.
    value_mapping: Dict[Any, Any] = None
    name_extended: str = None  # Optional longer description of feature.

    def fillna(self, data: pd.Series) -> pd.Series:
        """Apply the list of na_values, filling these values in data with np.nan."""
        logging.debug(f"replacing missing values of {self.na_values} "
                      f"for feature {self.name}")
        # Handles case where na values have misspecified type (i.e. float vs. int);
        # we would like the filling to be robust to these kinds of misspecification.
        if not isinstance(data.dtype, cat_dtype):
            try:
                na_values = np.array(self.na_values).astype(data.dtype)
            except ValueError:
                # Raised when some na_values cannot be cast to the target type
                na_values = np.array(self.na_values)
        else:
            na_values = np.array(self.na_values)
        return data.replace(na_values, np.nan)

    def apply_dtype(self, data: pd.Series) -> pd.Series:
        """Apply the specified dtype, casting if needed (and otherwise returning data)."""
        return safe_cast(data, self.kind)


@dataclass
class FeatureList:
    features: List[Feature]
    documentation: str = None  # optional link to docs

    @property
    def predictors(self) -> List[str]:
        """Fetch the names of non-target features."""
        return [x.name for x in self.features if not x.is_target]

    @property
    def names(self):
        return [f.name for f in self.features]

    @property
    def target(self):
        """Return the name of the target feature (if it exists)."""
        for f in self.features:
            if f.is_target:
                return f.name
        return None

    def __add__(self, other):
        if (self.target and other.target):
            raise ValueError("cannot add two lists which both contain targets.")
        return FeatureList(list(set(self.features + other.features)),
                           documentation=self.documentation)

    def __iter__(self):
        yield from self.features

    def apply_schema(self, df: pd.DataFrame,
                     passthrough_columns: Sequence[str] = None) -> pd.DataFrame:
        """Apply the schema defined in the FeatureList to the DataFrame.

        Subsets to only the columns corresponding to features in FeatureList,
        and then transforms each column by casting it to the type specified
        in each Feature.
        """
        if not passthrough_columns:
            passthrough_columns = []

        drop_cols = list(set(x for x in df.columns
                             if x not in self.names
                             and x not in passthrough_columns))
        if drop_cols:
            logging.info(f"dropping columns not in FeatureList: {drop_cols}")
            df.drop(columns=drop_cols, inplace=True)
        for f in self.features:
            logging.debug(f"checking feature {f.name}")
            if f.name not in df.columns:
                # Case: expected this feature, and it is missing.
                raise ValueError(f"feature {f.name} not present in data with"
                                 f"columns {df.columns}.")

            # Fill na values (before casting)
            if f.na_values:
                df[f.name] = f.fillna(df[f.name])

            # Cast to desired type
            if not column_is_of_type(df[f.name], f.kind):
                df[f.name] = f.apply_dtype(df[f.name])

        # Drop any rows containing missing values.
        if _contains_missing_values(df):
            logging.debug("missing values detected in data; counts by column:")
            logging.debug(pd.isnull(df).sum())

        return df


def _transformed_columns_to_numeric(df, prefix: str,
                                    to_type=float) -> pd.DataFrame:
    """Postprocess the results of a ColumnTransformer.

    ColumnTransformers convert their output to 'object' dtype, even when the
    outputs are properly numeric.

    Using pattern-matching from the verbose feature names of a
    ColumnTransformer, cast any transformed columns to the specified dtype.

    This provides maximum compatibility with downstream models by eliminating
    categorical dtypes where they are no longer needed (and no longer properly
    describe the data type of a column).

    Valid prefixes include "scale_" for scaled columns, and "onehot_" for
    one-hot-encoded columns.
    """
    cols_to_transform = [c for c in df.columns if c.startswith(prefix)]
    logging.debug(f"casting {len(cols_to_transform)} columns to type {to_type}")
    for c in tqdm(cols_to_transform):
        df[c] = df[c].astype(to_type)
    return df


@dataclass
class PreprocessorConfig:
    # Preprocessing for categorical features (also applies to boolean features).
    # Options are: one_hot, map_values, passthrough.
    categorical_features: str = "one_hot"
    # Preprocessing for float and int features.
    # Options: normalize, passthrough.
    numeric_features: str = "normalize"
    domain_labels: str = "label_encode"
    passthrough_columns: Union[
        str, List[str]] = None  # Feature names to passthrough, or "all".
    # If "rows", drop rows containing na values, if "columns", drop columns
    # containing na values; if None do not do anything for missing values.
    dropna: Union[str, None] = "rows"

    min_frequency: float = None  # see OneHotEncoder.min_frequency
    max_categories: int = None  # see OneHotEncoder.max_categories


def map_values(df: pd.DataFrame, mapping: dict) -> pd.DataFrame:
    return df.stack().map(mapping).unstack()


def get_numeric_columns(data: pd.DataFrame) -> List[str]:
    """Helper function to extract numeric colnames from a dataset."""
    numeric_columns = make_column_selector(
        pattern="^(?![Tt]arget)",
        dtype_include=np.number)(data)
    return numeric_columns


def get_categorical_columns(data: pd.DataFrame) -> List[str]:
    """Helper function to extract categorical colnames from a dataset."""
    categorical_columns = make_column_selector(
        pattern="^(?![Tt]arget)",
        dtype_include=[np.object, np.bool, cat_dtype])(data)
    return categorical_columns


def make_value_map_transforms(features_to_map: List[Feature]) -> List[
    Tuple[str, FunctionTransformer, List[str]]]:
    """Helper function to build the mapping transforms for a set of features.

    The output of this function can be used as input to a ColumnTransformer.
    """
    transforms = [
        (f.name,
         FunctionTransformer(partial(map_values,
                                     mapping=f.value_mapping),
                             check_inverse=False,
                             feature_names_out="one-to-one"),
         [f.name]) for f in features_to_map]
    return transforms


# TODO(jpgard): implement ability to apply mapper to categorical features.
@dataclass
class Preprocessor:
    config: PreprocessorConfig
    feature_transformer: ColumnTransformer = None
    domain_label_transformer: LabelEncoder = None
    feature_list: Optional[FeatureList] = None

    def _get_categorical_transforms(self, data: pd.DataFrame,
                                    passthrough_columns: List[str]) -> List:
        categorical_columns = get_categorical_columns(data)

        if self.config.categorical_features == "passthrough":
            transforms = []

        elif self.config.categorical_features == "one_hot":
            transforms = [
                (f'onehot_{c}',
                 OneHotEncoder(dtype=np.int8, categories=[data[c].unique()],
                               min_frequency=self.config.min_frequency,
                               max_categories=self.config.max_categories), [c])
                for c in categorical_columns
                if c not in passthrough_columns]

        elif self.config.categorical_features == "map_values":
            assert self.feature_list is not None
            features_to_map = [f for f in self.feature_list
                               if f.value_mapping is not None
                               and f.name in categorical_columns]
            assert len(features_to_map), \
                "No categorical columns with  mappings provided. Either provide " \
                "mappings for one or more columns or set " \
                "categorical_columns='passthrough' in the feature config. "
            transforms = make_value_map_transforms(features_to_map)

        else:
            raise ValueError(f"{self.config.categorical_features} is not "
                             "a valid categorical preprocessor type.")
        return transforms

    def _get_numeric_transforms(self, data: pd.DataFrame,
                                passthrough_columns: List[str] = None) -> List:
        numeric_columns = get_numeric_columns(data)
        cols = [c for c in numeric_columns if c not in passthrough_columns]
        if self.config.numeric_features == "passthrough":
            transforms = []

        elif self.config.numeric_features == "map_values":
            assert self.feature_list is not None
            features_to_map = [f for f in self.feature_list
                               if f.value_mapping is not None
                               and f.name in numeric_columns]
            assert len(features_to_map), \
                "No numeric columns with mappings provided. Either provide " \
                "mappings for one or more columns or set " \
                "numeric_columns='passthrough' in the feature config. "
            transforms = make_value_map_transforms(features_to_map)

        elif self.config.numeric_features == "normalize":
            transforms = [(f'scale_{c}', StandardScaler(), [c]) for c in cols]

        elif self.config.numeric_features == "kbins":
            transforms = [("kbin", KBinsDiscretizer(encode="ordinal"), cols)]

        else:
            raise ValueError(f"{self.config.numeric_features} is not "
                             f"a valid numeric preprocessor type.")
        return transforms

    def _post_transform_summary(self, data: pd.DataFrame):
        logging.debug("printing post-transform feature summary")
        if self.config.numeric_features == "kbins":
            for c in data.columns:
                if "kbin" in c: logging.info(f"{c}:{data[c].unique().tolist()}")
        elif self.config.numeric_features == "normalize":
            for c in data.columns:
                if "scale" in c: logging.info(f"{c}: mean {data[c].mean()}, "
                                              f"std {data[c].std()}")

    def fit_feature_transformer(self, data, train_idxs: List[int],
                                passthrough_columns: List[str] = None):
        """Fits the feature_transformer defined by this Preprocessor."""
        transforms = []
        transforms += self._get_numeric_transforms(data,
                                                   passthrough_columns)

        transforms += self._get_categorical_transforms(data,
                                                       passthrough_columns)

        self.feature_transformer = ColumnTransformer(
            transforms,
            remainder='passthrough',
            sparse_threshold=0,
            verbose_feature_names_out=False)

        self.feature_transformer.fit(data.loc[train_idxs, :])
        return

    def transform_features(self, data) -> pd.DataFrame:
        transformed = self.feature_transformer.transform(data)
        transformed = pd.DataFrame(
            transformed,
            columns=self.feature_transformer.get_feature_names_out())

        return transformed

    def _post_transform(self, transformed: pd.DataFrame,
                        cast_dtypes: Optional[Mapping] = None) -> pd.DataFrame:
        """Postprocess the result of a ColumnTransformer."""
        transformed.columns = [c.replace("remainder__", "")
                               for c in transformed.columns]
        transformed.columns = [sub_illegal_chars(c) for c in
                               transformed.columns]

        # By default transformed columns will be cast to 'object' dtype; we cast them
        # back to a numeric type.
        transformed = _transformed_columns_to_numeric(transformed, "onehot_",
                                                      np.int8)
        transformed = _transformed_columns_to_numeric(transformed, "scale_")
        # Cast the specified columns back to their original types
        if cast_dtypes:
            for colname, dtype in cast_dtypes.items():
                transformed[colname] = transformed[colname].astype(dtype)
        return transformed

    def _dropna(self, data: pd.DataFrame) -> pd.DataFrame:
        """Apply the specified handling of NA values.

        If "rows", drop any rows containing NA values. If "columns", drop
        any columns containing NA values. If None, do not alter data.

        This function should be called *before* splitting data.
        """
        if self.config.dropna is None:
            return data

        start_len = len(data)
        if self.config.dropna == "rows":
            data.dropna(inplace=True)
        elif self.config.dropna == "columns":
            data.dropna(axis=1, inplace=True)
        logging.debug(
            f"dropped {start_len - len(data)} rows "
            f"containing missing values "
            f"({(start_len - len(data)) * 100 / start_len}% of data).")
        data.reset_index(drop=True, inplace=True)
        if not len(data):
            raise ValueError(f"Data is empty after applying dropna="
                             f"{self.config.dropna}")
        return data

    def _check_inputs(self, data):
        prohibited_chars = "[].<>"
        for char in prohibited_chars:
            for colname in data.columns:
                if char in colname:
                    raise ValueError(
                        f"[ERROR] illegal character {char} in column name "
                        f"{colname}; this will likely lead to an error.")

    def fit_transform_domain_labels(self, x: pd.Series):
        if self.config.domain_labels == "label_encode":
            self.domain_label_transformer = LabelEncoder()
            return self.domain_label_transformer.fit_transform(x)
        else:
            raise NotImplementedError(f"Method {self.config.domain_labels} not "
                                      f"implemented.")

    def get_passthrough_columns(self, data: pd.DataFrame,
                                passthrough_columns: List[str] = None,
                                domain_label_colname: Optional[str] = None):
        if passthrough_columns is None:
            passthrough_columns = []

        if self.config.passthrough_columns:
            passthrough_columns += self.config.passthrough_columns

        if self.config.numeric_features == "passthrough":
            passthrough_columns += get_numeric_columns(data)

        if self.config.categorical_features == "passthrough":
            passthrough_columns += get_categorical_columns(data)

        if domain_label_colname and (
                domain_label_colname not in passthrough_columns):
            logging.debug(f"adding domain label column {domain_label_colname} "
                          f"to passthrough columns")
            passthrough_columns.append(domain_label_colname)
        return passthrough_columns

    def fit_transform(self, data: pd.DataFrame, train_idxs: List[int],
                      domain_label_colname: Optional[str] = None,
                      passthrough_columns: List[str] = None) -> pd.DataFrame:
        """Fit a feature_transformer and apply it to the input features."""
        logging.info(f"transforming columns")
        if self.config.passthrough_columns == "all":
            logging.info("passthrough is 'all'; data will not be preprocessed "
                         "by tableshift.")
            return data

        passthrough_columns = self.get_passthrough_columns(data,
                                                           passthrough_columns)

        # All non-domain label passthrough columns will be cast to their
        # original type post-transformation (ColumnTransformer
        # actually casts all columns to object type).
        dtypes_in = data.dtypes.to_dict()
        post_transform_cast_dtypes = (
            {c: dtypes_in[c] for c in passthrough_columns if
             c != domain_label_colname}
            if passthrough_columns else None)

        self._check_inputs(data)

        # Fit the feature transformer and apply it.
        self.fit_feature_transformer(data, train_idxs, passthrough_columns)
        transformed = self.transform_features(data)

        transformed = self._post_transform(
            transformed, cast_dtypes=post_transform_cast_dtypes)

        if domain_label_colname:
            # Case: fit the domain label transformer and apply it.
            transformed.loc[:,
            domain_label_colname] = self.fit_transform_domain_labels(
                transformed.loc[:, domain_label_colname])

        self._post_transform_summary(transformed)
        logging.info("transforming columns complete.")
        return transformed

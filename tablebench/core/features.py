from dataclasses import dataclass, field
import re
from typing import List, Any, Sequence, Optional, Mapping, Tuple, Union

import numpy as np
import pandas as pd
from pandas.api.types import CategoricalDtype as cat_dtype
from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.preprocessing import OneHotEncoder, StandardScaler, MinMaxScaler, \
    LabelEncoder
from tablebench.core.discretization import KBinsDiscretizer


def _contains_missing_values(df: pd.DataFrame) -> bool:
    return np.any(pd.isnull(df).values)


def safe_cast(x: pd.Series, dtype):
    print(f"[DEBUG] casting feature {x.name} from dtype "
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
                print(f"[WARNING] cannot cast feature {x.name} to "
                      f"dtype {dtype.__name__} due to missing values; "
                      f"attempting cast to float instead. Recommend changing"
                      f"the feature spec for this feature to type float.")
                return x.astype(float)


@dataclass(frozen=True)
class Feature:
    name: str
    kind: Any  # a class type to which the feature should be castable.
    description: str = None
    is_target: bool = False
    # Values, besides np.nan, to count as null/missing. These should be
    # values in the original feature encoding (that is, the values that would
    # occur for this column in the output of the preprocess_fn, not after
    # casting to `kind`), because values after casting may be unpredictable.
    na_values: Tuple = field(default_factory=tuple)


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
        self.features = list(set(self.features + other.features))
        return self

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

        def _column_is_of_type(x: pd.Series, dtype) -> bool:
            """Helper function to check whether column has specified dtype."""
            if hasattr(dtype, "name"):
                # Case: dtype is of categorical dtype; has a "name"
                # attribute identical to the pandas dtype name for
                # categorical data.
                return x.dtype.name == dtype.name
            else:
                return x.dtype == dtype.__name__

        drop_cols = list(set(x for x in df.columns
                             if x not in self.names
                             and x not in passthrough_columns))
        if drop_cols:
            print("[DEBUG] dropping data columns not in "
                  f"FeatureList: {drop_cols}")
            df.drop(columns=drop_cols, inplace=True)
        for f in self.features:
            print(f"[DEBUG] checking feature {f.name}")
            if f.name not in df.columns:
                # Case: expected this feature, and it is missing.
                raise ValueError(f"feature {f.name} not present in data with"
                                 f"columns {df.columns}.")

            # Fill na values (before casting)
            if f.na_values:
                print(f"[DEBUG] replacing missing values of {f.na_values} "
                      f"for feature {f.name}")

                df[f.name].replace(f.na_values, np.nan)

            # Cast to desired type
            if not _column_is_of_type(df[f.name], f.kind):
                df[f.name] = safe_cast(df[f.name], f.kind)

        # Drop any rows containing missing values.
        if _contains_missing_values(df):
            print("[DEBUG] missing values in data; counts by column:")
            print(pd.isnull(df).sum())

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
    for c in df.columns:
        if c.startswith(prefix):
            df[c] = df[c].astype(to_type)
    return df


# TODO(jpgard): separate the PreprocessorConfig from the Preprocessor class,
#  which should have a classmethod to instantiate Preprocessor from
#  PreprocessorConfig.
@dataclass
class PreprocessorConfig:
    categorical_features: str = "one_hot"  # also applies to boolean features.
    numeric_features: str = "normalize"
    domain_labels: str = "label_encode"
    feature_transformer: ColumnTransformer = None
    domain_label_transformer: LabelEncoder = None
    passthrough_columns: List[str] = None  # Feature names to passthrough.
    # If "rows", drop rows containing na values, if "columns", drop columns
    # containing na values; if None do not do anything for missing values.
    dropna: Union[str, None] = "rows"



    def _get_categorical_transforms(self, data: pd.DataFrame,
                                    categorical_columns: List[str],
                                    passthrough_columns: List[str]):
        if self.categorical_features == "one_hot":
            transforms = [
                (f'onehot_{c}',
                 OneHotEncoder(dtype='int', categories=[data[c].unique()]),
                 [c])
                for c in categorical_columns
                if c not in passthrough_columns]
        else:
            raise ValueError(f"{self.categorical_features} is not "
                             "a valid categorical preprocessor type.")
        return transforms

    def _get_numeric_transforms(self, numeric_columns: List[str],
                                passthrough_columns: List[str] = None):
        cols = [c for c in numeric_columns if c not in passthrough_columns]
        if self.numeric_features == "normalize":
            transforms = [(f'scale_{c}', StandardScaler(), [c]) for c in cols]
        elif self.numeric_features == "kbins":
            transforms = [("kbin", KBinsDiscretizer(encode="ordinal"), cols)]
        else:
            raise ValueError(f"{self.numeric_features} is not "
                             f"a valid numeric preprocessor type.")
        return transforms

    def _post_transform_summary(self, data: pd.DataFrame):
        print("[DEBUG] printing post-transform feature summary")
        if self.numeric_features == "kbins":
            for c in data.columns:
                if "kbin" in c: print(f"{c}: {data[c].unique().tolist()}")
        elif self.numeric_features == "normalize":
            for c in data.columns:
                if "scale" in c: print(f"{c}: mean {data[c].mean()}, "
                                       f"std {data[c].std()}")

    def fit_feature_transformer(self, data, train_idxs: List[int],
                                passthrough_columns: List[str] = None):
        """Fits the feature_transformer defined by this PreprocessorConfig."""

        numeric_columns = make_column_selector(
            pattern="^(?![Tt]arget)",
            dtype_include=np.number)(data)
        numeric_transforms = self._get_numeric_transforms(numeric_columns,
                                                          passthrough_columns)

        categorical_columns = make_column_selector(
            pattern="^(?![Tt]arget)",
            dtype_include=[np.object, np.bool, cat_dtype])(data)

        categorical_transforms = self._get_categorical_transforms(
            data, categorical_columns, passthrough_columns)

        transforms = numeric_transforms + categorical_transforms
        self.feature_transformer = ColumnTransformer(
            transforms,
            remainder='passthrough',
            sparse_threshold=0,
            verbose_feature_names_out=True)

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
        transformed.columns = [re.sub("[\\[\\].<>/,]", "", c) for c in
                               transformed.columns]

        transformed = _transformed_columns_to_numeric(transformed, "onehot_")
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
        if self.dropna is None:
            return data

        start_len = len(data)
        if self.dropna == "rows":
            data.dropna(inplace=True)
        elif self.dropna == "columns":
            data.dropna(axis=1, inplace=True)
        print(f"[DEBUG] dropped {start_len - len(data)} rows "
              f"containing missing values "
              f"({(start_len - len(data)) / start_len}% of data).")
        return data.reset_index(drop=True)

    def _check_inputs(self, data):
        prohibited_chars = "[].<>"
        for char in prohibited_chars:
            for colname in data.columns:
                if char in colname:
                    raise ValueError(
                        f"[ERROR] illegal character {char} in column name "
                        f"{colname}; this will likely lead to an error.")

    def fit_transform_domain_labels(self, x: pd.Series):
        if self.domain_labels == "label_encode":
            self.domain_label_transformer = LabelEncoder()
            return self.domain_label_transformer.fit_transform(x)
        else:
            raise NotImplementedError(f"Method {self.domain_labels} not "
                                      f"implemented.")

    def fit_transform(self, data: pd.DataFrame, train_idxs: List[int],
                      domain_label_colname: Optional[str],
                      passthrough_columns: List[str] = None) -> pd.DataFrame:
        """Fit a feature_transformer and apply it to the input features."""

        if self.passthrough_columns:
            passthrough_columns += self.passthrough_columns

        if domain_label_colname and domain_label_colname not in passthrough_columns:
            print(f"[DEBUG] adding domain label column {domain_label_colname} "
                  f"to passthrough columns")
            passthrough_columns.append(domain_label_colname)

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
            transformed.loc[:, domain_label_colname] = self.fit_transform_domain_labels(
                transformed.loc[:, domain_label_colname])

        self._post_transform_summary(transformed)
        return transformed

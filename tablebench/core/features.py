from dataclasses import dataclass
import re
from typing import List, Any, Sequence, Optional, Mapping

import numpy as np
import pandas as pd
from pandas.api.types import CategoricalDtype as cat_dtype
from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.preprocessing import OneHotEncoder, StandardScaler, MinMaxScaler
from tablebench.core.discretization import KBinsDiscretizer


def safe_cast(x: pd.Series, dtype):
    if dtype == cat_dtype:
        return x.apply(str).astype("category")
    else:
        return x.astype(dtype)


@dataclass(frozen=True)
class Feature:
    name: str
    kind: Any  # a class type to which the feature should be castable.
    description: str = None
    is_target: bool = False


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
            if not _column_is_of_type(df[f.name], f.kind):
                print(f"[DEBUG] casting feature {f.name} from dtype "
                      f"{df[f.name].dtype.name} to dtype {f.kind.__name__}")
                df[f.name] = safe_cast(df[f.name], f.kind)
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


@dataclass
class PreprocessorConfig:
    categorical_features: str = "one_hot"  # also applies to boolean features.
    numeric_features: str = "normalize"
    transformer: ColumnTransformer = None

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

    def fit_transformer(self, data, train_idxs: List[int],
                        passthrough_columns: List[str] = None):
        """Fits the transformer associated with this PreprocessorConfig."""

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
        self.transformer = ColumnTransformer(
            transforms,
            remainder='passthrough',
            sparse_threshold=0,
            verbose_feature_names_out=True)

        self.transformer.fit(data.loc[train_idxs, :])
        return

    def transform(self, data) -> pd.DataFrame:
        transformed = self.transformer.transform(data)
        transformed = pd.DataFrame(
            transformed,
            columns=self.transformer.get_feature_names_out())

        return transformed

    def _post_transform(self, transformed: pd.DataFrame,
                        cast_dtypes: Optional[Mapping] = None) -> pd.DataFrame:
        """Postprocess the result of a ColumnTransformer."""
        transformed.columns = [c.replace("remainder__", "")
                               for c in transformed.columns]
        transformed.columns = [re.sub("[\\[\\].<>]", "", c) for c in
                               transformed.columns]

        transformed = _transformed_columns_to_numeric(transformed, "onehot_")
        transformed = _transformed_columns_to_numeric(transformed, "scale_")
        # Cast the specified columns back to their original types
        if cast_dtypes:
            for colname, dtype in cast_dtypes.items():
                transformed[colname] = transformed[colname].astype(dtype)
        return transformed

    def fit_transform(self, data: pd.DataFrame, train_idxs: List[int],
                      passthrough_columns: List[str] = None) -> pd.DataFrame:
        dtypes_in = data.dtypes.to_dict()
        passthrough_dtypes = ({c: dtypes_in[c] for c in
                               passthrough_columns}
                              if passthrough_columns
                              else None)
        self.fit_transformer(data, train_idxs, passthrough_columns)
        transformed = self.transform(data)
        transformed = self._post_transform(transformed,
                                           cast_dtypes=passthrough_dtypes)
        self._post_transform_summary(transformed)
        return transformed

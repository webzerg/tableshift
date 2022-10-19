from dataclasses import dataclass
from typing import List, Any

import numpy as np
import pandas as pd
from pandas.api.types import CategoricalDtype as cat_dtype
from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.preprocessing import OneHotEncoder, StandardScaler, MinMaxScaler


def safe_cast(x: pd.Series, dtype):
    if dtype == cat_dtype:
        return x.apply(str).astype("category")
    else:
        return x.astype(dtype)


@dataclass(frozen=True)
class Feature:
    name: str
    kind: Any  # a type to which the feature should be castable.
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
                     allow_missing_cols: List = None) -> pd.DataFrame:
        """Apply the schema defined in the FeatureList to the DataFrame.

        Subsets to only the columns corresponding to features in FeatureList,
        and then transforms each column by casting it to the type specified
        in each Feature.
        """
        drop_cols = list(set(x for x in df.columns if x not in self.names))
        if drop_cols:
            print("[DEBUG] dropping data columns not in "
                  f"FeatureList: {drop_cols}")
            df.drop(columns=drop_cols, inplace=True)
        for f in self.features:
            if f.name not in df.columns and f.name not in allow_missing_cols:
                # Case: expected this feature, and it is missing.
                raise ValueError(f"feature {f.name} not present in data with"
                                 f"columns {df.columns}.")
            elif f.name not in df.columns:
                # Case: this feature is missing, but it is allowed to be missing
                # (e.g. it is a domain-split feature, which was used to split
                # the data and then is removed).
                continue
            if not df[f.name].dtype == f.kind:
                print(f"[INFO] casting feature {f.name} from type "
                      f"{df[f.name].dtype} to dtype {f.kind}")
                df[f.name] = safe_cast(df[f.name], f.kind)
        return df


@dataclass
class PreprocessorConfig:
    categorical_features: str = "one_hot"  # also applies to boolean features.
    numeric_features: str = "normalize"
    transformer: ColumnTransformer = None

    def fit_transformer(self, data, train_idxs: List[int],
                        passthrough_columns: List[str] = None):
        """Fits the transformer associated with this PreprocessorConfig."""

        numeric_columns = make_column_selector(
            pattern="^(?![Tt]arget)",
            dtype_include=np.number)(data)
        numeric_transforms = [
            (f'scale_{c}', StandardScaler(), [c])
            for c in numeric_columns
            if c not in passthrough_columns]

        categorical_columns = make_column_selector(
            pattern="^(?![Tt]arget)",
            dtype_include=[np.object, np.bool, cat_dtype])(data)

        categorical_transforms = [
            (f'onehot_{c}',
             OneHotEncoder(dtype='int', categories=[data[c].unique()]),
             [c])
            for c in categorical_columns
            if c not in passthrough_columns]

        transforms = numeric_transforms + categorical_transforms
        self.transformer = ColumnTransformer(
            transforms,
            remainder='passthrough',
            sparse_threshold=0,
            verbose_feature_names_out=False)

        self.transformer.fit(data.loc[train_idxs, :])
        return

    def transform(self, data) -> pd.DataFrame:
        transformed = self.transformer.transform(data)
        transformed = pd.DataFrame(
            transformed,
            columns=self.transformer.get_feature_names_out())
        transformed.columns = [c.replace("remainder__", "")
                               for c in transformed.columns]
        return transformed

    def fit_transform(self, data, train_idxs: List[int],
                      passthrough_columns: List[str] = None):
        self.fit_transformer(data, train_idxs, passthrough_columns)
        return self.transform(data)

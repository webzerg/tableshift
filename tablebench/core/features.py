from dataclasses import dataclass
from typing import List, Any

import numpy as np
import pandas as pd
from pandas.api.types import CategoricalDtype as cat_dtype
from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.preprocessing import OneHotEncoder, StandardScaler, MinMaxScaler


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

    def apply_schema(self, df: pd.DataFrame):
        """Apply the schema defined in the FeatureList to the DataFrame.

        Subsets to only the columns corresponding to features in FeatureList,
        and then transforms each column by casting it to the type specified
        in each Feature.
        """
        raise NotImplementedError


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

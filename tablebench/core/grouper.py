from typing import Mapping, Sequence, Any, List
from dataclasses import dataclass

import pandas as pd
from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.preprocessing import FunctionTransformer


@dataclass
class Grouper:
    features_and_values: Mapping[str, Sequence[Any]]
    drop: bool
    transformer: ColumnTransformer = None

    @property
    def features(self) -> List[str]:
        return list(self.features_and_values.keys())

    def transform(self, data) -> pd.DataFrame:
        for c in self.features:
            data[c] = data[c].apply(
                lambda x: int(x in self.features_and_values[c]))
        return data

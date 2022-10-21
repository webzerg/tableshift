"""Utilities and constants for the Adult dataset."""
import pandas as pd

from tablebench.core.features import Feature, FeatureList, cat_dtype

ADULT_RESOURCES = [
    "https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data",
    "https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.names",
    "https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.test",
]

# Names to use for the features in the Adult dataset. These correspond to
# (human-readable) column names in the order of the columns in adult.data file.
ADULT_FEATURE_NAMES = ["Age", "Workclass", "fnlwgt", "Education",
                       "Education-Num",
                       "Marital Status",
                       "Occupation", "Relationship", "Race", "Sex",
                       "Capital Gain",
                       "Capital Loss",
                       "Hours per week", "Country", "Target"]

ADULT_FEATURES = FeatureList(features=[
    Feature("Age", float),
    Feature("Workclass", cat_dtype),
    Feature("Education-Num", cat_dtype),
    Feature("Marital Status", cat_dtype),
    Feature("Occupation", cat_dtype),
    Feature("Relationship", cat_dtype),
    Feature("Race", cat_dtype),
    Feature("Sex", cat_dtype),
    Feature("Capital Gain", float),
    Feature("Capital Loss", float),
    Feature("Hours per week", float),
    Feature("Country", cat_dtype),
    Feature("Target", int, is_target=True),
])


def preprocess_adult(df: pd.DataFrame):
    """Process a raw adult dataset."""
    df['Target'] = df['Target'].replace(
        {'<=50K': 0,
         '<=50K.': 0,
         '>50K': 1,
         '>50K.': 1})
    del df['Education']
    return df

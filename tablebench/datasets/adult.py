"""Utilities and constants for the Adult dataset."""
import pandas as pd

from tablebench.core.features import Feature, FeatureList

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
    Feature("Age", int),
    Feature("Workclass", str),
    Feature("fnlwgt", int),
    Feature("Education-Num", str),
    Feature("Marital Status", str),
    Feature("Occupation", str),
    Feature("Relationship", str),
    Feature("Race", str),
    Feature("Sex", str),
    Feature("Capital Gain", int),
    Feature("Capital Loss", int),
    Feature("Hours per week", int),
    Feature("Country", str),
    Feature("Target", int),
])


def preprocess_adult(df: pd.DataFrame):
    """Process a raw adult dataset."""
    df['Target'] = df['Target'].replace(
        {'<=50K': 0,
         '>50K': 1,
         '<=50K.': 0,
         '>50K.': 1})
    del df['Education']
    return df

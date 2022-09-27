import pandas as pd

from tablebench.core.features import Feature, FeatureList

GERMAN_RESOURCES = [
    "https://archive.ics.uci.edu/ml/machine-learning-databases/"
    "statlog/german/german.data"
]

GERMAN_FEATURES = FeatureList(features=[
    Feature("status", str),
    Feature("duration", int),
    Feature("credit_history", str),
    Feature("purpose", str),
    Feature("credit_amt", int),
    Feature("savings_acct_bonds", str),
    Feature("present_unemployed_since", str),
    Feature("installment_rate", int),
    Feature("other_debtors", str),
    Feature("pres_res_since", int),
    Feature("property", str),
    Feature("age", int),
    Feature("other_installment", str),
    Feature("housing", str),
    Feature("num_exist_credits", int),
    Feature("job", str),
    Feature("num_ppl", int),
    Feature("has_phone", str),
    Feature("foreign_worker", str),
    Feature("Target", int)])


def preprocess_german(df: pd.DataFrame):
    df.columns = ["status", "duration", "credit_history",
                  "purpose", "credit_amt", "savings_acct_bonds",
                  "present_unemployed_since", "installment_rate",
                  "per_status_sex", "other_debtors", "pres_res_since",
                  "property", "age", "other_installment", "housing",
                  "num_exist_credits", "job", "num_ppl", "has_phone",
                  "foreign_worker", "Target"]
    # Code labels as in tfds; see
    # https://github.com/tensorflow/datasets/blob/master/"\
    # "tensorflow_datasets/structured/german_credit_numeric.py
    df["Target"] = 2 - df["Target"]
    # convert per_status_sex into separate columns.
    # Sex is 1 if male; else 0.
    df["sex"] = df["per_status_sex"].apply(
        lambda x: 1 if x not in ["A92", "A95"] else 0)
    # Age is 1 if above median age, else 0.
    median_age = df["age"].median()
    df["age"] = df["age"].apply(lambda x: 1 if x > median_age else 0)

    df["single"] = df["per_status_sex"].apply(
        lambda x: 1 if x in ["A93", "A95"] else 0)

    df.drop(columns="per_status_sex", inplace=True)
    return df

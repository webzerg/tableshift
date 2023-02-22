import pandas as pd

from tableshift.core.features import Feature, FeatureList, cat_dtype

GERMAN_RESOURCES = [
    "https://archive.ics.uci.edu/ml/machine-learning-databases/"
    "statlog/german/german.data"
]

GERMAN_FEATURES = FeatureList(features=[
    Feature("status", cat_dtype),
    Feature("duration", float),
    Feature("credit_history", cat_dtype),
    Feature("purpose", cat_dtype),
    Feature("credit_amt", float),
    Feature("savings_acct_bonds", cat_dtype),
    Feature("present_unemployed_since", cat_dtype),
    Feature("installment_rate", float),
    Feature("other_debtors", cat_dtype),
    Feature("pres_res_since", float),
    Feature("property", cat_dtype),
    Feature("age_geq_median", cat_dtype),
    Feature("sex", cat_dtype),
    Feature("other_installment", cat_dtype),
    Feature("housing", cat_dtype),
    Feature("num_exist_credits", float),
    Feature("job", cat_dtype),
    Feature("num_ppl", float),
    Feature("has_phone", cat_dtype),
    Feature("foreign_worker", cat_dtype),
    Feature("Target", int, is_target=True)])


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
    df["age_geq_median"] = df["age"].apply(lambda x: 1 if x > median_age else 0)

    df["single"] = df["per_status_sex"].apply(
        lambda x: 1 if x in ["A93", "A95"] else 0)

    df.drop(columns=["per_status_sex", "age"], inplace=True)
    return df

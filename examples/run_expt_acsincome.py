"""
An example script to train a model on the ACS Income dataset.

This example shows how to use multiple years from an ACS data source, using
data from both 2017 and 2018, and to use this to evaluate temporal shift.

Usage:
    python run_expt.py
"""
from tablebench.core import DomainSplitter, Grouper, TabularDataset, \
    TabularDatasetConfig, PreprocessorConfig

from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import accuracy_score

dataset_config = TabularDatasetConfig()

preprocessor_config = PreprocessorConfig()

splitter = DomainSplitter(val_size=0.01, random_state=956523, eval_size=0.5,
                          domain_split_varname="ACS_YEAR",
                          domain_split_ood_values=[2018],
                          domain_split_id_values=[2017])

grouper = Grouper({"RAC1P": [1, ], "SEX": [1, ]}, drop=False)
acsincome = TabularDataset("acsincome",
                           acs_task="acsincome",
                           config=dataset_config,
                           splitter=splitter,
                           grouper=grouper,
                           preprocessor_config=preprocessor_config,
                           years=(2017, 2018))

X_tr, y_tr, G_tr = acsincome.get_pandas(split="train")

estimator = HistGradientBoostingClassifier()

print(f"fitting estimator of type {type(estimator)}")
estimator.fit(X_tr, y_tr)
print("fitting estimator complete.")

X_te, y_te, G_te = acsincome.get_pandas(split="test")

y_hat_te = estimator.predict(X_te)
test_accuracy = accuracy_score(y_te, y_hat_te)
print(f"test accuracy is: {test_accuracy:.3f}")

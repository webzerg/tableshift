"""
An example script to train a model on the physionet challenge 2019 dataset.

Usage:
    python examples/run_expt_physionet.py
"""
from tablebench.core import DomainSplitter, \
    Grouper, TabularDataset, \
    TabularDatasetConfig, PreprocessorConfig

from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import accuracy_score

dataset_config = TabularDatasetConfig()

preprocessor_config = PreprocessorConfig()

splitter = DomainSplitter(val_size=0.05,
                          eval_size=0.2,
                          random_state=43406,
                          domain_split_varname="set",
                          domain_split_ood_values=["a"])
grouper = Grouper({"Age": [1, ], "Gender": [1, ]}, drop=False)
dset = TabularDataset("physionet",
                      config=dataset_config,
                      splitter=splitter,
                      grouper=grouper,
                      preprocessor_config=preprocessor_config)

X_tr, y_tr, G_tr = dset.get_pandas(split="train")

estimator = HistGradientBoostingClassifier()
estimator.fit(X_tr, y_tr)

X_te, y_te, G_te = dset.get_pandas(split="id_test")

y_hat_te = estimator.predict(X_te)
test_accuracy = accuracy_score(y_te, y_hat_te)
print(f"in-domain test accuracy is: {test_accuracy:.3f}")

X_te, y_te, G_te = dset.get_pandas(split="ood_test")

y_hat_te = estimator.predict(X_te)
test_accuracy = accuracy_score(y_te, y_hat_te)
print(f"out-of-domain test accuracy is: {test_accuracy:.3f}")

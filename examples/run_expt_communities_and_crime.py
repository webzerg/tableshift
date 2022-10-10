"""
An example script to train a model on the German credit dataset.

Usage:
    python examples/run_expt_german.py
"""
from tablebench.core import RandomSplitter, DomainSplitter, \
    Grouper, TabularDataset, \
    TabularDatasetConfig, PreprocessorConfig

from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import accuracy_score

dataset_config = TabularDatasetConfig()

preprocessor_config = PreprocessorConfig()

splitter = RandomSplitter(test_size=0.2, val_size=0.05, random_state=94427)

grouper = Grouper({"Race": [1, ], "income_level_above_median": [1, ]},
                  drop=False)
dset = TabularDataset("communities_and_crime",
                      config=dataset_config,
                      splitter=splitter,
                      grouper=grouper,
                      preprocessor_config=preprocessor_config)

X_tr, y_tr, G_tr = dset.get_pandas(split="train")

estimator = HistGradientBoostingClassifier()
print(f"fitting estimator of type {type(estimator)}")
estimator.fit(X_tr, y_tr)
print("fitting estimator complete.")

X_te, y_te, G_te = dset.get_pandas(split="test")

y_hat_te = estimator.predict(X_te)
test_accuracy = accuracy_score(y_te, y_hat_te)
print(f"test accuracy is: {test_accuracy:.3f}")

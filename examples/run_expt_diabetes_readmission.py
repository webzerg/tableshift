"""
An example script to train a model on the Adult dataset.

Usage:
    python run_expt.py
"""
from tablebench.core import RandomSplitter, Grouper, TabularDataset, \
    TabularDatasetConfig, PreprocessorConfig

from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import accuracy_score

dataset_config = TabularDatasetConfig()

preprocessor_config = PreprocessorConfig()

splitter = RandomSplitter(test_size=0.25, val_size=0.01, random_state=29746)
# male vs. all others; white non-hispanic vs. others
grouper = Grouper({"race": ["Caucasian", ], "gender": ["Male", ]}, drop=False)
dset = TabularDataset("diabetes_readmission",
                      config=dataset_config,
                      splitter=splitter,
                      grouper=grouper,
                      preprocessor_config=preprocessor_config)

X_tr, y_tr, G_tr = dset.get_pandas(split="train")

estimator = HistGradientBoostingClassifier()

print(f"fitting estimator of type {type(estimator)} with shape {X_tr.shape}")
estimator.fit(X_tr, y_tr)
print("fitting estimator complete.")

X_te, y_te, G_te = dset.get_pandas(split="test")

y_hat_te = estimator.predict(X_te)
test_accuracy = accuracy_score(y_te, y_hat_te)
print(f"test accuracy is: {test_accuracy:.3f}")

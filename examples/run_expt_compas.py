"""
An example script to train a model on the COMPAS dataset.

Usage:
    python run_expt.py
"""
from tablebench.core import RandomSplitter, Grouper, TabularDataset, \
    TabularDatasetConfig, PreprocessorConfig

from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import accuracy_score

dataset_config = TabularDatasetConfig(
    cache_dir="./tmp",
    download=True,
    random_seed=12334)

preprocessor_config = PreprocessorConfig()

splitter = RandomSplitter(test_size=0.2, val_size=0.05, random_state=90127)
grouper = Grouper({"race": ["Caucasian", ], "sex": ["Male", ]}, drop=False)
compas = TabularDataset("compas",
                        config=dataset_config,
                        splitter=splitter,
                        grouper=grouper,
                        preprocessor_config=preprocessor_config)

X_tr, y_tr, G_tr = compas.get_pandas(split="train")

estimator = HistGradientBoostingClassifier()
estimator.fit(X_tr, y_tr)

X_te, y_te, G_te = compas.get_pandas(split="test")

y_hat_te = estimator.predict(X_te)
test_accuracy = accuracy_score(y_te, y_hat_te)
print(f"test accuracy is: {test_accuracy:.3f}")

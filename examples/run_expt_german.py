"""
An example script to train a model on the German credit dataset.

Usage:
    python examples/run_expt_german.py
"""
from tablebench.core import RandomSplitter, DomainSplitter,\
    Grouper, TabularDataset, \
    TabularDatasetConfig, PreprocessorConfig

from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import accuracy_score

dataset_config = TabularDatasetConfig()

preprocessor_config = PreprocessorConfig()

# splitter = RandomSplitter(test_size=0.1, val_size=0.05, random_state=90127)
splitter = DomainSplitter(val_size=0.05,
                          random_state=12406,
                          domain_split_varname="purpose",
                          domain_split_ood_values=["A41", "A42", "A43"])
grouper = Grouper({"sex": [1, ], "age": [1, ]}, drop=False)
german = TabularDataset("german",
                        config=dataset_config,
                        splitter=splitter,
                        grouper=grouper,
                        preprocessor_config=preprocessor_config)

X_tr, y_tr, G_tr = german.get_pandas(split="train")
import ipdb;ipdb.set_trace()
estimator = HistGradientBoostingClassifier()
estimator.fit(X_tr, y_tr)

X_te, y_te, G_te = german.get_pandas(split="test")

y_hat_te = estimator.predict(X_te)
test_accuracy = accuracy_score(y_te, y_hat_te)
print(f"test accuracy is: {test_accuracy:.3f}")

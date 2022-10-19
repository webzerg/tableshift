"""
An example script to train a model on the German credit dataset.

Usage:
    python examples/run_expt_german.py
"""
from pprint import pprint

from tablebench.core import RandomSplitter, DomainSplitter, \
    Grouper, TabularDataset, \
    TabularDatasetConfig, PreprocessorConfig

from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import accuracy_score

dataset_config = TabularDatasetConfig()

preprocessor_config = PreprocessorConfig()

# splitter = RandomSplitter(test_size=0.1, val_size=0.05, random_state=90127)
splitter = DomainSplitter(val_size=0.05,
                          eval_size=0.2,
                          random_state=12406,
                          domain_split_varname="purpose",
                          domain_split_ood_values=["A41", "A42", "A43"])
grouper = Grouper({"sex": [1, ], "age_>=_median": [1, ]}, drop=False)
dset = TabularDataset("german",
                      config=dataset_config,
                      splitter=splitter,
                      grouper=grouper,
                      preprocessor_config=preprocessor_config)

X_tr, y_tr, G_tr = dset.get_pandas(split="train")

estimator = HistGradientBoostingClassifier()
estimator.fit(X_tr, y_tr)

for split in ("id_test", "ood_test"):

    X_te, _, _ = dset.get_pandas(split=split)

    y_hat_te = estimator.predict(X_te)
    metrics = dset.evaluate_predictions(y_hat_te, split=split)
    print(f"metrics on split {split}:")
    for k, v in metrics.items():
        print(f"\t{k:<40}:{v:.3f}")

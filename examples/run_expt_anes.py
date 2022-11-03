"""
An example script to train a model on the ANES dataset.

This script illustrates the use of DomainSplitter to specify a subdomain for
training (i.e. train on 2004-2016), and a separate one for testing (test on
2020).

Usage:
    python run_expt.py
"""
from tablebench.core import DomainSplitter, Grouper, TabularDataset, \
    TabularDatasetConfig, PreprocessorConfig

from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import accuracy_score

dataset_config = TabularDatasetConfig()

preprocessor_config = PreprocessorConfig(numeric_features="kbins")

splitter = DomainSplitter(val_size=0.01, eval_size=0.2, random_state=45345,
                          domain_split_varname="VCF0004",
                          domain_split_ood_values=[2020],
                          domain_split_id_values=[2004, 2008, 2012, 2016])
# male vs. all others; white non-hispanic vs. others
grouper = Grouper({"VCF0104": ["1", ], "VCF0105a": ["1.0", ]}, drop=False)
dset = TabularDataset("anes",
                      config=dataset_config,
                      splitter=splitter,
                      grouper=grouper,
                      preprocessor_config=preprocessor_config)

X_tr, y_tr, G_tr = dset.get_pandas(split="train")

estimator = HistGradientBoostingClassifier()

print(f"fitting estimator of type {type(estimator)} with shape {X_tr.shape}")
estimator.fit(X_tr, y_tr)
print("fitting estimator complete.")

for split in ("id_test", "ood_test"):
    X_te, y_te, G_te = dset.get_pandas(split=split)
    print(X_te.shape)
    y_hat_te = estimator.predict(X_te)
    test_accuracy = accuracy_score(y_te, y_hat_te)
    print(f"accuracy on split {split} is: {test_accuracy:.3f}")

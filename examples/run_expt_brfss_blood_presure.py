"""
An example script to train a model on the Adult dataset.

Usage:
    python run_expt.py
"""
import argparse
from tablebench.core import RandomSplitter, Grouper, TabularDataset, \
    TabularDatasetConfig, PreprocessorConfig
from tablebench.datasets.brfss import BRFSS_YEARS

from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import accuracy_score


def main(cache_dir):
    dataset_config = TabularDatasetConfig(cache_dir=cache_dir)

    preprocessor_config = PreprocessorConfig(passthrough_columns=["IYEAR"])

    splitter = RandomSplitter(test_size=0.5, val_size=0.25, random_state=29746)
    grouper = Grouper({"PRACE1": [1, ], "SEX": [1, ]}, drop=False)
    dset = TabularDataset("brfss_blood_pressure",
                          years=BRFSS_YEARS,
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


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache_dir", default="tmp",
                        help="Directory to cache raw data files to.")
    args = parser.parse_args()
    main(**vars(args))

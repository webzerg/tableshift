import argparse
from dataclasses import dataclass
from datetime import datetime
from typing import Sequence, Any

import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegressionCV
import xgboost as xgb

from tablebench.core import DomainSplitter, Grouper, TabularDataset, \
    TabularDatasetConfig, PreprocessorConfig
from tablebench.datasets.acs import ACS_STATE_LIST
from tablebench.datasets.brfss import BRFSS_STATE_LIST

preprocessor_config = PreprocessorConfig()


class XGB(xgb.XGBClassifier):
    def __init__(self, **kwargs):
        super().__init__(self, **kwargs, enable_categorical=True)


estimator_cls = (LogisticRegressionCV,
                 HistGradientBoostingClassifier,
                 XGB)


@dataclass
class DomainShiftExperimentConfig:
    tabular_dataset_kwargs: dict
    domain_split_varname: str
    domain_split_ood_values: Sequence[Any]
    grouper: Grouper
    dataset_config: TabularDatasetConfig


# Set of fixed domain shift experiments.
experiment_configs = {
    "acsincome_st": DomainShiftExperimentConfig(
        tabular_dataset_kwargs={"name": "acsincome",
                                "acs_task": "acsincome"},
        domain_split_varname="ST",
        domain_split_ood_values=ACS_STATE_LIST,
        grouper=Grouper({"RAC1P": [1, ], "SEX": [1, ]}, drop=False),
        dataset_config=TabularDatasetConfig()),

    "acspubcov_st": DomainShiftExperimentConfig(
        tabular_dataset_kwargs={"name": "acspubcov",
                                "acs_task": "acspubcov"},
        domain_split_varname="ST",
        domain_split_ood_values=ACS_STATE_LIST,
        grouper=Grouper({"RAC1P": [1, ], "SEX": [1, ]}, drop=False),
        dataset_config=TabularDatasetConfig()),

    "brfss_st": DomainShiftExperimentConfig(
        tabular_dataset_kwargs={"name": "brfss"},
        domain_split_varname="STATE",
        domain_split_ood_values=BRFSS_STATE_LIST,
        grouper=Grouper({"PRACE1": [1, ], "SEX": [1, ]}, drop=False),
        dataset_config=TabularDatasetConfig()),

    # "anes_st": DomainShiftExperimentConfig(
    #     tabular_dataset_kwargs={"name": "anes"},
    #     domain_split_varname="STATE",
    #     domain_split_ood_values=BRFSS_STATE_LIST,
    #     grouper=Grouper({"RIDRETH3": ["3", ], "RIAGENDR": ["1", ]}, drop=False),
    #     dataset_config=TabularDatasetConfig()),
}


def main(experiment):
    iterates = []

    experiment_config = experiment_configs[experiment]
    for tgt in experiment_config.domain_split_ood_values:

        splitter = DomainSplitter(
            val_size=0.01,
            eval_size=1 / 5.,
            domain_split_varname=experiment_config.domain_split_varname,
            domain_split_ood_values=[tgt],
            random_state=19542)

        dset = TabularDataset(
            **experiment_config.tabular_dataset_kwargs,
            config=experiment_config.dataset_config,
            splitter=splitter,
            grouper=experiment_config.grouper,
            preprocessor_config=preprocessor_config)

        X_tr, y_tr, G_tr = dset.get_pandas(split="train")

        for est_cls in estimator_cls:
            estimator = est_cls()

            print(f"fitting estimator of type {type(estimator)} with "
                  f"target {experiment_config.domain_split_varname} = {tgt}")
            estimator.fit(X_tr, y_tr)
            print("fitting estimator complete.")

            metrics = {"estimator": str(type(estimator)),
                       "task": experiment_config.tabular_dataset_kwargs[
                           "name"],
                       "domain_split_varname": experiment_config.domain_split_varname,
                       "domain_split_odd_values": tgt,
                       }
            for split in ("id_test", "ood_test"):
                X_te, _, _ = dset.get_pandas(split=split)

                y_hat_te = estimator.predict(X_te)
                split_metrics = dset.evaluate_predictions(y_hat_te,
                                                          split=split)
                metrics.update(split_metrics)
            iterates.append(metrics)

    results = pd.DataFrame(iterates)
    results.to_csv(
        f"results-{experiment}-{str(datetime.now()).replace(' ', '')}.csv")
    return


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment", choices=list(experiment_configs.keys()))
    args = parser.parse_args()
    main(**vars(args))

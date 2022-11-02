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
from tablebench.datasets.anes import ANES_STATES
from tablebench.datasets.brfss import BRFSS_STATE_LIST
from tablebench.datasets.communities_and_crime import CANDC_STATE_LIST

estimator_cls = (LogisticRegressionCV,
                 HistGradientBoostingClassifier,
                 xgb.XGBClassifier)


@dataclass
class DomainShiftExperimentConfig:
    tabular_dataset_kwargs: dict
    domain_split_varname: str
    domain_split_ood_values: Sequence[Any]
    grouper: Grouper
    dataset_config: TabularDatasetConfig
    preprocessor_config: PreprocessorConfig


# Set of fixed domain shift experiments.
experiment_configs = {
    "acsincome_st": DomainShiftExperimentConfig(
        tabular_dataset_kwargs={"name": "acsincome",
                                "acs_task": "acsincome"},
        domain_split_varname="ST",
        domain_split_ood_values=ACS_STATE_LIST,
        grouper=Grouper({"RAC1P": [1, ], "SEX": [1, ]}, drop=False),
        dataset_config=TabularDatasetConfig(),
        preprocessor_config=PreprocessorConfig()),

    "acspubcov_st": DomainShiftExperimentConfig(
        tabular_dataset_kwargs={"name": "acspubcov",
                                "acs_task": "acspubcov"},
        domain_split_varname="ST",
        domain_split_ood_values=ACS_STATE_LIST,
        grouper=Grouper({"RAC1P": [1, ], "SEX": [1, ]}, drop=False),
        dataset_config=TabularDatasetConfig(),
        preprocessor_config=PreprocessorConfig()),

    "brfss_st": DomainShiftExperimentConfig(
        tabular_dataset_kwargs={"name": "brfss"},
        domain_split_varname="STATE",
        domain_split_ood_values=BRFSS_STATE_LIST,
        grouper=Grouper({"PRACE1": [1, ], "SEX": [1, ]}, drop=False),
        dataset_config=TabularDatasetConfig(),
        preprocessor_config=PreprocessorConfig()),

    "candc_st": DomainShiftExperimentConfig(
        tabular_dataset_kwargs={"name": "communities_and_crime"},
        domain_split_varname="state",
        domain_split_ood_values=CANDC_STATE_LIST,
        grouper=Grouper({"Race": [1, ], "income_level_above_median": [1, ]},
                        drop=False),
        dataset_config=TabularDatasetConfig(),
        preprocessor_config=PreprocessorConfig(),
    ),

    "diabetes_admtype": DomainShiftExperimentConfig(
        tabular_dataset_kwargs={"name": "diabetes_readmission"},
        domain_split_varname='admission_type_id',
        domain_split_ood_values=[1, 2, 3, 4, 5, 6, 7, 8],
        grouper=Grouper({"race": ["Caucasian", ], "gender": ["Male", ]},
                        drop=False),
        dataset_config=TabularDatasetConfig(),
        preprocessor_config=PreprocessorConfig(),
    ),

    "diabetes_admsrc": DomainShiftExperimentConfig(
        tabular_dataset_kwargs={"name": "diabetes_readmission"},
        domain_split_varname='admission_source_id',
        domain_split_ood_values=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 14, 17,
                                 20, 22, 25],
        grouper=Grouper({"race": ["Caucasian", ], "gender": ["Male", ]},
                        drop=False),
        dataset_config=TabularDatasetConfig(),
        preprocessor_config=PreprocessorConfig(),
    ),

    "mooc_course": DomainShiftExperimentConfig(
        tabular_dataset_kwargs={"name": "mooc"},
        domain_split_varname="course_id",
        domain_split_ood_values=['HarvardX/CB22x/2013_Spring',
                                 'HarvardX/CS50x/2012',
                                 'HarvardX/ER22x/2013_Spring',
                                 'HarvardX/PH207x/2012_Fall',
                                 'HarvardX/PH278x/2013_Spring'],
        grouper=Grouper({"gender": ["m", ],
                         "LoE_DI": ["Bachelor's", "Master's", "Doctorate"]},
                        drop=False),
        dataset_config=TabularDatasetConfig(),
        preprocessor_config=PreprocessorConfig(),
    ),

    "physionet_set": DomainShiftExperimentConfig(
        tabular_dataset_kwargs={"name": "physionet"},
        domain_split_varname="set",
        domain_split_ood_values=["a", "b"],
        grouper=Grouper({"Age": [x for x in range(40, 100)], "Gender": [1, ]},
                        drop=False),
        dataset_config=TabularDatasetConfig(),
        preprocessor_config=PreprocessorConfig(numeric_features="kbins")
    ),

    "anes_st": DomainShiftExperimentConfig(
        tabular_dataset_kwargs={"name": "anes", "years": [2020, ]},
        domain_split_varname="VCF0901b",
        domain_split_ood_values=ANES_STATES,
        grouper=Grouper({"VCF0104": ["1", ], "VCF0105a": ["1.0", ]},
                        drop=False),
        dataset_config=TabularDatasetConfig(),
        preprocessor_config=PreprocessorConfig()),
}


def main(experiment):
    iterates = []

    expt_config = experiment_configs[experiment]
    for tgt in expt_config.domain_split_ood_values:

        splitter = DomainSplitter(
            val_size=0.01,
            eval_size=1 / 5.,
            domain_split_varname=expt_config.domain_split_varname,
            domain_split_ood_values=[tgt],
            random_state=19542)

        try:
            dset = TabularDataset(
                **expt_config.tabular_dataset_kwargs,
                config=expt_config.dataset_config,
                splitter=splitter,
                grouper=expt_config.grouper,
                preprocessor_config=expt_config.preprocessor_config)
        except ValueError as ve:
            # Case: split is too small.
            print(f"[WARNING] error initializing dataset for expt {experiment} "
                  f"with {expt_config.domain_split_varname} == {tgt}: {ve}")
            continue

        X_tr, y_tr, G_tr = dset.get_pandas(split="train")

        for est_cls in estimator_cls:
            estimator = est_cls()

            print(f"fitting estimator of type {type(estimator)} with "
                  f"target {expt_config.domain_split_varname} = {tgt}")
            estimator.fit(X_tr, y_tr)
            print("fitting estimator complete.")

            metrics = {"estimator": str(type(estimator)),
                       "task": expt_config.tabular_dataset_kwargs[
                           "name"],
                       "domain_split_varname": expt_config.domain_split_varname,
                       "domain_split_ood_values": tgt,
                       }
            for split in ("id_test", "ood_test"):
                try:
                    X_te, _, _ = dset.get_pandas(split=split)

                    y_hat_te = estimator.predict(X_te)
                    split_metrics = dset.evaluate_predictions(y_hat_te,
                                                              split=split)
                    metrics.update(split_metrics)
                except Exception as e:
                    print(f"exception evaluating split {split} with "
                          f"{expt_config.domain_split_varname}=={tgt}: "
                          f"{e}; skipping")
                    continue
            iterates.append(metrics)

    results = pd.DataFrame(iterates)
    results.to_csv(
        f"results-{experiment}-{str(datetime.now()).replace(' ', '')}.csv")
    return


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment", choices=list(experiment_configs.keys()),
                        default="brfss_st")
    args = parser.parse_args()
    main(**vars(args))

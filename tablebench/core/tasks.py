"""
Contains task configurations.

A task is a set of features (including both predictors and a target variable)
along with a DataSource. Tasks are the fundamental benchmarks that comprise
the tableshift benchmark.
"""

from dataclasses import dataclass
from typing import Any
from .data_source import *
from .features import FeatureList

from tablebench.datasets import *


@dataclass
class TaskConfig:
    # The data_source_cls instantiates the DataSource,
    # which fetches data and preprocesses it using a preprocess_fn.
    data_source_cls: Any
    # The feature_list describes the schema of the data *after* the
    # preprocess_fn is applied. It is used to check the output of the
    # preprocess_fn, and features are dropped or type-cast as necessary.
    feature_list: FeatureList


# Mapping of task names to their configs. An arbitrary number of tasks
# can be created from a single data source, by specifying different
# preprocess_fn and features.
_TASK_REGISTRY = {
    "acsincome": TaskConfig(ACSDataSource,
                            ACS_INCOME_FEATURES + ACS_SHARED_FEATURES),
    "acsfoodstamps": TaskConfig(ACSDataSource,
                                ACS_FOODSTAMPS_FEATURES + ACS_SHARED_FEATURES),
    "acspubcov": TaskConfig(ACSDataSource,
                            ACS_PUBCOV_FEATURES + ACS_SHARED_FEATURES),
    "acsunemployment": TaskConfig(
        ACSDataSource,
        ACS_UNEMPLOYMENT_FEATURES + ACS_SHARED_FEATURES),
    "adult": TaskConfig(AdultDataSource, ADULT_FEATURES),
    "anes": TaskConfig(ANESDataSource, ANES_FEATURES),
    "brfss": TaskConfig(BRFSSDataSource, BRFSS_FEATURES),
    "communities_and_crime": TaskConfig(CommunitiesAndCrimeDataSource,
                                        CANDC_FEATURES),
    "compas": TaskConfig(COMPASDataSource, COMPAS_FEATURES),
    "diabetes_readmission": TaskConfig(DiabetesReadmissionDataSource,
                                       DIABETES_READMISSION_FEATURES),
    "german": TaskConfig(GermanDataSource, GERMAN_FEATURES),
    "mooc": TaskConfig(MOOCDataSource, MOOC_FEATURES),
    "nhanes_cholesterol": TaskConfig(NHANESDataSource,
                                     NHANES_CHOLESTEROL_FEATURES + \
                                     NHANES_DEMOG_FEATURES),
    "physionet": TaskConfig(PhysioNetDataSource, PHYSIONET_FEATURES),
}


def get_task_config(name: str) -> TaskConfig:
    if name in _TASK_REGISTRY:
        return _TASK_REGISTRY[name]
    else:
        raise NotImplementedError(f"task {name} not implemented.")

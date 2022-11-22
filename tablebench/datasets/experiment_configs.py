from dataclasses import dataclass

from tablebench.core import RandomSplitter, Grouper, PreprocessorConfig, \
    DomainSplitter, FixedSplitter, Splitter
from tablebench.datasets import BRFSS_YEARS


@dataclass
class ExperimentConfig:
    splitter: Splitter
    grouper: Grouper
    preprocessor_config: PreprocessorConfig
    tabular_dataset_kwargs: dict


EXPERIMENT_CONFIGS = {
    "acsfoodstamps": ExperimentConfig(
        splitter=RandomSplitter(test_size=0.5, val_size=0.25,
                                random_state=29746),
        grouper=Grouper({"RAC1P": [1, ], "SEX": [1, ]}, drop=False),
        preprocessor_config=PreprocessorConfig(),
        tabular_dataset_kwargs={"acs_task": "acsfoodstamps"}),

    "acsincome": ExperimentConfig(
        splitter=DomainSplitter(val_size=0.01, random_state=956523,
                                id_test_size=0.5,
                                domain_split_varname="ACS_YEAR",
                                domain_split_ood_values=[2018],
                                domain_split_id_values=[2017]),
        grouper=Grouper({"RAC1P": [1, ], "SEX": [1, ]}, drop=False),
        preprocessor_config=PreprocessorConfig(),
        tabular_dataset_kwargs={"acs_task": "acsfoodstamps",
                                "years": (2017, 2018)}),

    "acspubcov": ExperimentConfig(
        splitter=RandomSplitter(test_size=0.5, val_size=0.25,
                                random_state=29746),
        grouper=Grouper({"RAC1P": [1, ], "SEX": [1, ]}, drop=False),
        preprocessor_config=PreprocessorConfig(),
        tabular_dataset_kwargs={"acs_task": "acspubcov"}),

    "acsunemployment": ExperimentConfig(
        splitter=RandomSplitter(test_size=0.5, val_size=0.25,
                                random_state=29746),
        grouper=Grouper({"RAC1P": [1, ], "SEX": [1, ]}, drop=False),
        preprocessor_config=PreprocessorConfig(),
        tabular_dataset_kwargs={"acs_task": "acsunemployment"}),

    "adult": ExperimentConfig(
        splitter=FixedSplitter(val_size=0.25, random_state=29746),
        grouper=Grouper({"Race": ["White", ], "Sex": ["Male", ]}, drop=False),
        preprocessor_config=PreprocessorConfig(), tabular_dataset_kwargs={}),

    "anes": ExperimentConfig(
        splitter=DomainSplitter(val_size=0.01, id_test_size=0.2,
                                random_state=45345,
                                domain_split_varname="VCF0004",
                                domain_split_ood_values=[2020],
                                domain_split_id_values=[2004, 2008, 2012,
                                                        2016]),
        # male vs. all others; white non-hispanic vs. others
        grouper=Grouper({"VCF0104": ["1", ], "VCF0105a": ["1.0", ]},
                        drop=False),
        preprocessor_config=PreprocessorConfig(numeric_features="kbins",
                                               dropna=None),
        tabular_dataset_kwargs={}),

    "brfss_blood_pressure": ExperimentConfig(
        splitter=RandomSplitter(test_size=0.5, val_size=0.25,
                                random_state=29746),
        grouper=Grouper({"PRACE1": [1, ], "SEX": [1, ]}, drop=False),
        preprocessor_config=PreprocessorConfig(passthrough_columns=["IYEAR"]),
        tabular_dataset_kwargs={"years": BRFSS_YEARS},
    ),

    "brfss_diabetes": ExperimentConfig(
        splitter=RandomSplitter(test_size=0.5, val_size=0.25,
                                random_state=29746),
        grouper=Grouper({"PRACE1": [1, ], "SEX": [1, ]}, drop=False),
        preprocessor_config=PreprocessorConfig(passthrough_columns=["IYEAR"]),
        tabular_dataset_kwargs={"years": BRFSS_YEARS},
    ),

    "communities_and_crime": ExperimentConfig(
        splitter=RandomSplitter(test_size=0.2, val_size=0.01,
                                random_state=94427),
        grouper=Grouper({"Race": [1, ], "income_level_above_median": [1, ]},
                        drop=False),
        preprocessor_config=PreprocessorConfig(), tabular_dataset_kwargs={}),

    "compas": ExperimentConfig(
        splitter=RandomSplitter(test_size=0.2, val_size=0.01,
                                random_state=90127),
        grouper=Grouper({"race": ["Caucasian", ], "sex": ["Male", ]},
                        drop=False),
        preprocessor_config=PreprocessorConfig(), tabular_dataset_kwargs={}),

    "diabetes_readmission": ExperimentConfig(
        splitter=RandomSplitter(test_size=0.25, val_size=0.01,
                                random_state=29746),
        # male vs. all others; white non-hispanic vs. others
        grouper=Grouper({"race": ["Caucasian", ], "gender": ["Male", ]},
                        drop=False),
        preprocessor_config=PreprocessorConfig(), tabular_dataset_kwargs={}),

    "german": ExperimentConfig(
        splitter=RandomSplitter(val_size=0.01, test_size=0.2, random_state=832),
        grouper=Grouper({"sex": ['1', ], "age_geq_median": ['1', ]},
                        drop=False),
        preprocessor_config=PreprocessorConfig(), tabular_dataset_kwargs={}),

    "mooc": ExperimentConfig(
        splitter=DomainSplitter(val_size=0.01,
                                id_test_size=0.2,
                                random_state=43406,
                                domain_split_varname="course_id",
                                domain_split_ood_values=[
                                    "HarvardX/CB22x/2013_Spring"]),
        grouper=Grouper({"gender": ["m", ],
                         "LoE_DI": ["Bachelor's", "Master's", "Doctorate"]},
                        drop=False),
        preprocessor_config=PreprocessorConfig(), tabular_dataset_kwargs={}),

    "nhanes": ExperimentConfig(
        splitter=RandomSplitter(test_size=0.5, val_size=0.25,
                                random_state=29746),
        # Race (non. hispanic white vs. all others; male vs. all others)
        grouper=Grouper({"RIDRETH3": ["3.0", ], "RIAGENDR": ["1.0", ]},
                        drop=False),
        preprocessor_config=PreprocessorConfig(
            passthrough_columns=["nhanes_year"],
            numeric_features="kbins"),
        tabular_dataset_kwargs={}),

    "physionet": ExperimentConfig(
        splitter=DomainSplitter(val_size=0.05,
                                id_test_size=0.2,
                                random_state=43406,
                                domain_split_varname="set",
                                domain_split_ood_values=["a"]),
        grouper=Grouper({"Age": [x for x in range(40, 100)], "Gender": [1, ]},
                        drop=False),
        preprocessor_config=PreprocessorConfig(numeric_features="kbins"),
        tabular_dataset_kwargs={})
}
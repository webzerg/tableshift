from dataclasses import dataclass
from typing import Sequence, Optional, Any
from tablebench.core import Grouper, PreprocessorConfig
from tablebench.core.utils import sliding_window
from tablebench.datasets import ACS_STATE_LIST, ACS_YEARS, BRFSS_STATE_LIST, \
    BRFSS_YEARS, CANDC_STATE_LIST, NHANES_YEARS, ANES_STATES, ANES_YEARS


@dataclass
class DomainShiftExperimentConfig:
    tabular_dataset_kwargs: dict
    domain_split_varname: str
    domain_split_ood_values: Sequence[Any]
    grouper: Grouper
    preprocessor_config: PreprocessorConfig
    domain_split_id_values: Optional[Sequence[Any]] = None


# Set of fixed domain shift experiments.
domain_shift_experiment_configs = {
    "acsfoodstamps_st": DomainShiftExperimentConfig(
        tabular_dataset_kwargs={"name": "acsfoodstamps",
                                "acs_task": "acsfoodstamps"},
        domain_split_varname="ST",
        domain_split_ood_values=ACS_STATE_LIST,
        grouper=Grouper({"RAC1P": [1, ], "SEX": [1, ]}, drop=False),
        preprocessor_config=PreprocessorConfig()),

    "acsfoodstamps_year": DomainShiftExperimentConfig(
        tabular_dataset_kwargs={"name": "acsfoodstamps",
                                "acs_task": "acsfoodstamps",
                                "years": ACS_YEARS},
        domain_split_varname="ACS_YEAR",
        domain_split_ood_values=[ACS_YEARS[i + 1] for i in
                                 range(len(ACS_YEARS) - 1)],
        domain_split_id_values=[[ACS_YEARS[i]] for i in
                                range(len(ACS_YEARS) - 1)],
        grouper=Grouper({"RAC1P": [1, ], "SEX": [1, ]}, drop=False),
        preprocessor_config=PreprocessorConfig()),

    "acsincome_st": DomainShiftExperimentConfig(
        tabular_dataset_kwargs={"name": "acsincome",
                                "acs_task": "acsincome"},
        domain_split_varname="ST",
        domain_split_ood_values=ACS_STATE_LIST,
        grouper=Grouper({"RAC1P": [1, ], "SEX": [1, ]}, drop=False),
        preprocessor_config=PreprocessorConfig()),

    "acsincome_year": DomainShiftExperimentConfig(
        tabular_dataset_kwargs={"name": "acsincome",
                                "acs_task": "acsincome",
                                "years": ACS_YEARS},
        domain_split_varname="ACS_YEAR",
        domain_split_ood_values=[ACS_YEARS[i + 1] for i in
                                 range(len(ACS_YEARS) - 1)],
        domain_split_id_values=[[ACS_YEARS[i]] for i in
                                range(len(ACS_YEARS) - 1)],
        grouper=Grouper({"RAC1P": [1, ], "SEX": [1, ]}, drop=False),
        preprocessor_config=PreprocessorConfig()),

    "acspubcov_st": DomainShiftExperimentConfig(
        tabular_dataset_kwargs={"name": "acspubcov",
                                "acs_task": "acspubcov"},
        domain_split_varname="ST",
        domain_split_ood_values=ACS_STATE_LIST,
        grouper=Grouper({"RAC1P": [1, ], "SEX": [1, ]}, drop=False),
        preprocessor_config=PreprocessorConfig()),

    "acspubcov_year": DomainShiftExperimentConfig(
        tabular_dataset_kwargs={"name": "acspubcov",
                                "acs_task": "acspubcov",
                                "years": ACS_YEARS},
        domain_split_varname="ACS_YEAR",
        domain_split_ood_values=[ACS_YEARS[i + 1] for i in
                                 range(len(ACS_YEARS) - 1)],
        domain_split_id_values=[[ACS_YEARS[i]] for i in
                                range(len(ACS_YEARS) - 1)],
        grouper=Grouper({"RAC1P": [1, ], "SEX": [1, ]}, drop=False),
        preprocessor_config=PreprocessorConfig()),

    "acsunemployment_st": DomainShiftExperimentConfig(
        tabular_dataset_kwargs={"name": "acsunemployment",
                                "acs_task": "acsunemployment"},
        domain_split_varname="ST",
        domain_split_ood_values=ACS_STATE_LIST,
        grouper=Grouper({"RAC1P": [1, ], "SEX": [1, ]}, drop=False),
        preprocessor_config=PreprocessorConfig()),

    "acsunemployment_year": DomainShiftExperimentConfig(
        tabular_dataset_kwargs={"name": "acsunemployment",
                                "acs_task": "acsunemployment",
                                "years": ACS_YEARS},
        domain_split_varname="ACS_YEAR",
        domain_split_ood_values=[ACS_YEARS[i + 1] for i in
                                 range(len(ACS_YEARS) - 1)],
        domain_split_id_values=[[ACS_YEARS[i]] for i in
                                range(len(ACS_YEARS) - 1)],
        grouper=Grouper({"RAC1P": [1, ], "SEX": [1, ]}, drop=False),
        preprocessor_config=PreprocessorConfig()),

    "brfss_diabetes_st": DomainShiftExperimentConfig(
        tabular_dataset_kwargs={"name": "brfss_diabetes"},
        domain_split_varname="STATE",
        domain_split_ood_values=BRFSS_STATE_LIST,
        grouper=Grouper({"PRACE1": [1, ], "SEX": [1, ]}, drop=False),
        preprocessor_config=PreprocessorConfig()),

    "brfss_diabetes_year": DomainShiftExperimentConfig(
        tabular_dataset_kwargs={"name": "brfss_diabetes", "years": BRFSS_YEARS},
        domain_split_varname="IYEAR",
        domain_split_ood_values=[BRFSS_YEARS[i + 1] for i in
                                 range(len(BRFSS_YEARS) - 1)],
        domain_split_id_values=[[BRFSS_YEARS[i]] for i in
                                range(len(BRFSS_YEARS) - 1)],
        grouper=Grouper({"PRACE1": [1, ], "SEX": [1, ]}, drop=False),
        preprocessor_config=PreprocessorConfig()),
    "brfss_blood_pressure_st": DomainShiftExperimentConfig(
        tabular_dataset_kwargs={"name": "brfss_blood_pressure"},
        domain_split_varname="STATE",
        domain_split_ood_values=BRFSS_STATE_LIST,
        grouper=Grouper({"PRACE1": [1, ], "SEX": [1, ]}, drop=False),
        preprocessor_config=PreprocessorConfig()),

    "brfss_blood_pressure_year": DomainShiftExperimentConfig(
        tabular_dataset_kwargs={"name": "brfss_blood_pressure",
                                "years": BRFSS_YEARS},
        domain_split_varname="IYEAR",
        domain_split_ood_values=[BRFSS_YEARS[i + 1] for i in
                                 range(len(BRFSS_YEARS) - 1)],
        domain_split_id_values=[[BRFSS_YEARS[i]] for i in
                                range(len(BRFSS_YEARS) - 1)],
        grouper=Grouper({"PRACE1": [1, ], "SEX": [1, ]}, drop=False),
        preprocessor_config=PreprocessorConfig()),

    "candc_st": DomainShiftExperimentConfig(
        tabular_dataset_kwargs={"name": "communities_and_crime"},
        domain_split_varname="state",
        domain_split_ood_values=CANDC_STATE_LIST,
        grouper=Grouper({"Race": [1, ], "income_level_above_median": [1, ]},
                        drop=False),
        preprocessor_config=PreprocessorConfig(),
    ),

    "diabetes_admtype": DomainShiftExperimentConfig(
        tabular_dataset_kwargs={"name": "diabetes_readmission"},
        domain_split_varname='admission_type_id',
        domain_split_ood_values=[1, 2, 3, 4, 5, 6, 7, 8],
        grouper=Grouper({"race": ["Caucasian", ], "gender": ["Male", ]},
                        drop=False),
        preprocessor_config=PreprocessorConfig(),
    ),

    "diabetes_admsrc": DomainShiftExperimentConfig(
        tabular_dataset_kwargs={"name": "diabetes_readmission"},
        domain_split_varname='admission_source_id',
        domain_split_ood_values=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 14, 17,
                                 20, 22, 25],
        grouper=Grouper({"race": ["Caucasian", ], "gender": ["Male", ]},
                        drop=False),
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
        preprocessor_config=PreprocessorConfig(),
    ),

    "nhanes_year": DomainShiftExperimentConfig(
        tabular_dataset_kwargs={"name": "nhanes_cholesterol"},
        domain_split_varname="nhanes_year",
        domain_split_ood_values=[NHANES_YEARS[i + 1] for i in
                                 range(len(NHANES_YEARS) - 1)],
        domain_split_id_values=[[NHANES_YEARS[i]] for i in
                                range(len(NHANES_YEARS) - 1)],
        grouper=Grouper({"RIDRETH3": ["3.0", ], "RIAGENDR": ["1.0", ]},
                        drop=False),
        preprocessor_config=PreprocessorConfig(numeric_features="kbins"),

    ),

    "physionet_set": DomainShiftExperimentConfig(
        tabular_dataset_kwargs={"name": "physionet"},
        domain_split_varname="set",
        domain_split_ood_values=["a", "b"],
        grouper=Grouper({"Age": [x for x in range(40, 100)], "Gender": [1, ]},
                        drop=False),
        preprocessor_config=PreprocessorConfig(numeric_features="kbins")
    ),

    "anes_st": DomainShiftExperimentConfig(
        tabular_dataset_kwargs={"name": "anes", "years": [2020, ]},
        domain_split_varname="VCF0901b",
        domain_split_ood_values=ANES_STATES,
        grouper=Grouper({"VCF0104": ["1", ], "VCF0105a": ["1.0", ]},
                        drop=False),
        preprocessor_config=PreprocessorConfig(numeric_features="kbins")),

    "anes_year": DomainShiftExperimentConfig(
        tabular_dataset_kwargs={"name": "anes"},
        domain_split_varname="VCF0004",
        domain_split_ood_values=ANES_YEARS[4:],
        domain_split_id_values=list(sliding_window(ANES_YEARS, 4)),
        grouper=Grouper({"VCF0104": ["1", ], "VCF0105a": ["1.0", ]},
                        drop=False),
        preprocessor_config=PreprocessorConfig(numeric_features="kbins")),
}
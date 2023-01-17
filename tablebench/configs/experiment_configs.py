from dataclasses import dataclass

from tablebench.core import RandomSplitter, Grouper, PreprocessorConfig, \
    DomainSplitter, FixedSplitter, Splitter
from tablebench.datasets import BRFSS_YEARS, ANES_YEARS, ACS_YEARS, NHANES_YEARS
from tablebench.datasets.mimic_extract_feature_lists import \
    MIMIC_EXTRACT_SHARED_FEATURES
from tablebench.datasets.mimic_extract import MIMIC_EXTRACT_STATIC_FEATURES
from tablebench.configs.experiment_defaults import DEFAULT_ID_TEST_SIZE, \
    DEFAULT_OOD_VAL_SIZE, DEFAULT_ID_VAL_SIZE, DEFAULT_RANDOM_STATE


@dataclass
class ExperimentConfig:
    splitter: Splitter
    grouper: Grouper
    preprocessor_config: PreprocessorConfig
    tabular_dataset_kwargs: dict


# We passthrough all non-static columns because we use
# MIMIC-extract's default preprocessing/imputation and do not
# wish to modify it for these features
# (static features are not preprocessed by MIMIC-extract). See
# tableshift.datasets.mimic_extract.preprocess_mimic_extract().
_MIMIC_EXTRACT_PASSTHROUGH_COLUMNS = [
    f for f in MIMIC_EXTRACT_SHARED_FEATURES.names
    if f not in MIMIC_EXTRACT_STATIC_FEATURES.names]

EXPERIMENT_CONFIGS = {
    "acsfoodstamps": ExperimentConfig(
        splitter=DomainSplitter(val_size=DEFAULT_ID_VAL_SIZE,
                                ood_val_size=DEFAULT_OOD_VAL_SIZE,
                                random_state=DEFAULT_RANDOM_STATE,
                                id_test_size=DEFAULT_ID_TEST_SIZE,
                                domain_split_varname="DIVISION",
                                domain_split_ood_values=['06']),
        grouper=Grouper({"RAC1P": [1, ], "SEX": [1, ]}, drop=False),
        preprocessor_config=PreprocessorConfig(),
        tabular_dataset_kwargs={"acs_task": "acsfoodstamps"}),

    "acsincome": ExperimentConfig(
        splitter=DomainSplitter(val_size=DEFAULT_ID_VAL_SIZE,
                                ood_val_size=DEFAULT_OOD_VAL_SIZE,
                                random_state=DEFAULT_RANDOM_STATE,
                                id_test_size=DEFAULT_ID_TEST_SIZE,
                                domain_split_varname="DIVISION",
                                domain_split_ood_values=['01']),
        grouper=Grouper({"RAC1P": [1, ], "SEX": [1, ]}, drop=False),
        preprocessor_config=PreprocessorConfig(),
        tabular_dataset_kwargs={"acs_task": "acsincome"}),

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

    # ANES, Split by region; OOD is south: (AL, AR, DE, D.C., FL, GA, KY, LA,
    # MD, MS, NC, OK, SC,TN, TX, VA, WV)
    "anes": ExperimentConfig(
        splitter=DomainSplitter(val_size=DEFAULT_ID_VAL_SIZE,
                                ood_val_size=DEFAULT_OOD_VAL_SIZE,
                                random_state=DEFAULT_RANDOM_STATE,
                                id_test_size=DEFAULT_ID_TEST_SIZE,
                                domain_split_varname='VCF0112',
                                domain_split_ood_values=['3.0']),
        # male vs. all others; white non-hispanic vs. others
        grouper=Grouper({"VCF0104": ["1", ], "VCF0105a": ["1.0", ]},
                        drop=False),
        preprocessor_config=PreprocessorConfig(numeric_features="kbins",
                                               dropna=None),
        tabular_dataset_kwargs={}),

    "brfss_blood_pressure": ExperimentConfig(
        splitter=DomainSplitter(val_size=0.1, id_test_size=0.1,
                                random_state=29746,
                                domain_split_varname="POVERTY",
                                domain_split_ood_values=[1],
                                domain_split_id_values=[0, ]),
        grouper=Grouper({"PRACE1": [1, ], "SEX": [1, ]}, drop=False),
        preprocessor_config=PreprocessorConfig(passthrough_columns=["IYEAR"]),
        tabular_dataset_kwargs={"task": "blood_pressure", "years": BRFSS_YEARS},
    ),

    # "White nonhispanic" (in-domain) vs. all other race/ethnicity codes (OOD)
    "brfss_diabetes": ExperimentConfig(
        splitter=DomainSplitter(val_size=DEFAULT_ID_VAL_SIZE,
                                ood_val_size=DEFAULT_OOD_VAL_SIZE,
                                random_state=DEFAULT_RANDOM_STATE,
                                id_test_size=DEFAULT_ID_TEST_SIZE,
                                domain_split_varname="PRACE1",
                                domain_split_ood_values=[2, 3, 4, 5, 6],
                                domain_split_id_values=[1, ]),
        grouper=Grouper({"SEX": [1, ]}, drop=False),
        preprocessor_config=PreprocessorConfig(passthrough_columns=["IYEAR"]),
        tabular_dataset_kwargs={"name": "brfss_diabetes",
                                "task": "diabetes", "years": BRFSS_YEARS},
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

    "_debug": ExperimentConfig(
        splitter=DomainSplitter(
            val_size=0.01,
            id_test_size=0.2,
            ood_val_size=0.25,
            random_state=43406,
            domain_split_varname="purpose",
            # Counts by domain are below. We hold out all of the smallest
            # domains to avoid errors with very small domains during dev.
            # A48       9
            # A44      12
            # A410     12
            # A45      22
            # A46      50
            # A49      97
            # A41     103
            # A42     181
            # A40     234
            # A43     280
            domain_split_ood_values=["A44", "A410", "A45", "A46", "A48"]
        ),
        grouper=Grouper({"sex": ['1', ], "age_geq_median": ['1', ]},
                        drop=False),
        preprocessor_config=PreprocessorConfig(),
        tabular_dataset_kwargs={"name": "german"}),
    "diabetes_readmission": ExperimentConfig(
        splitter=DomainSplitter(val_size=DEFAULT_ID_VAL_SIZE,
                                ood_val_size=DEFAULT_OOD_VAL_SIZE,
                                random_state=DEFAULT_RANDOM_STATE,
                                id_test_size=DEFAULT_ID_TEST_SIZE,
                                domain_split_varname='admission_source_id',
                                domain_split_ood_values=[7, ]),
        # male vs. all others; white non-hispanic vs. others
        grouper=Grouper({"race": ["Caucasian", ], "gender": ["Male", ]},
                        drop=False),
        # Note: using min_frequency=0.01 reduces data
        # dimensionality from ~2400 -> 169 columns.
        # This is due to high cardinality of 'diag_*' features.
        preprocessor_config=PreprocessorConfig(min_frequency=0.01),
        tabular_dataset_kwargs={}),

    "german": ExperimentConfig(
        splitter=RandomSplitter(val_size=0.01, test_size=0.2, random_state=832),
        grouper=Grouper({"sex": ['1', ], "age_geq_median": ['1', ]},
                        drop=False),
        preprocessor_config=PreprocessorConfig(), tabular_dataset_kwargs={}),
    "heloc": ExperimentConfig(
        splitter=DomainSplitter(
            val_size=0.01,
            id_test_size=0.1,
            random_state=43590,
            domain_split_varname='NetFractionRevolvingBurdenPercentile',
            domain_split_ood_values=list(range(90, 101))),
        grouper=None,
        preprocessor_config=PreprocessorConfig(),
        tabular_dataset_kwargs={"name": "heloc"},
    ),
    "mimic_extract_los_3": ExperimentConfig(
        splitter=DomainSplitter(val_size=DEFAULT_ID_VAL_SIZE,
                                ood_val_size=DEFAULT_OOD_VAL_SIZE,
                                random_state=DEFAULT_RANDOM_STATE,
                                id_test_size=DEFAULT_ID_TEST_SIZE,
                                domain_split_varname="insurance",
                                domain_split_ood_values=["Medicare"]),

        grouper=Grouper({"gender": ['M'], }, drop=False),
        preprocessor_config=PreprocessorConfig(
            passthrough_columns=_MIMIC_EXTRACT_PASSTHROUGH_COLUMNS),
        tabular_dataset_kwargs={"task": "los_3"}),

    "mimic_extract_hosp_mort": ExperimentConfig(
        splitter=DomainSplitter(val_size=DEFAULT_ID_VAL_SIZE,
                                ood_val_size=DEFAULT_OOD_VAL_SIZE,
                                random_state=DEFAULT_RANDOM_STATE,
                                id_test_size=DEFAULT_ID_TEST_SIZE,
                                domain_split_varname="insurance",
                                domain_split_ood_values=["Medicare",
                                                         "Medicaid"]),
        grouper=Grouper({"gender": ['M'], }, drop=False),
        preprocessor_config=PreprocessorConfig(
            passthrough_columns=_MIMIC_EXTRACT_PASSTHROUGH_COLUMNS),
        tabular_dataset_kwargs={"task": "mort_hosp"}),

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

    "nhanes_cholesterol": ExperimentConfig(
        splitter=DomainSplitter(
            val_size=0.1,
            random_state=453879,
            id_test_size=0.1,
            domain_split_varname='RIDRETH_merged',
            domain_split_ood_values=[1, 2, 4, 6, 7],
            domain_split_id_values=[3],
        ),
        # Group by male vs. all others
        grouper=Grouper({"RIAGENDR": ["1.0", ]}, drop=False),
        preprocessor_config=PreprocessorConfig(
            passthrough_columns=["nhanes_year"],
            numeric_features="kbins"),
        tabular_dataset_kwargs={"nhanes_task": "cholesterol",
                                "years": NHANES_YEARS}),

    "nhanes_lead": ExperimentConfig(
        splitter=DomainSplitter(
            val_size=0.1,
            random_state=53079340,
            id_test_size=0.1,
            domain_split_varname='INDFMPIRBelowCutoff',
            domain_split_ood_values=[1.]),
        # Race (non. hispanic white vs. all others; male vs. all others)
        grouper=Grouper({"RIDRETH_merged": [3, ], "RIAGENDR": ["1.0", ]},
                        drop=False),
        preprocessor_config=PreprocessorConfig(
            passthrough_columns=["nhanes_year"],
            numeric_features="kbins"),
        tabular_dataset_kwargs={"nhanes_task": "lead", "years": NHANES_YEARS}),

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

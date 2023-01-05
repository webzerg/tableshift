from dataclasses import dataclass

from tablebench.core import RandomSplitter, Grouper, PreprocessorConfig, \
    DomainSplitter, FixedSplitter, Splitter
from tablebench.datasets import BRFSS_YEARS, ANES_YEARS, ACS_YEARS
from tablebench.datasets.mimic_extract_feature_lists import \
    MIMIC_EXTRACT_SHARED_FEATURES
from tablebench.datasets.mimic_extract import MIMIC_EXTRACT_STATIC_FEATURES


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
                                domain_split_ood_values=[ACS_YEARS[-1]],
                                domain_split_id_values=ACS_YEARS[:-1]),
        grouper=Grouper({"RAC1P": [1, ], "SEX": [1, ]}, drop=False),
        preprocessor_config=PreprocessorConfig(),
        tabular_dataset_kwargs={"acs_task": "acsincome",
                                "years": ACS_YEARS}),

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
                                domain_split_ood_values=[ANES_YEARS[-1]],
                                domain_split_id_values=ANES_YEARS[:-1]),
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
            domain_split_ood_values=["A40", "A41", "A42", "A44", "A410",
                                  "A45", "A46", "A48"]
        ),
        grouper=Grouper({"sex": ['1', ], "age_geq_median": ['1', ]},
                        drop=False),
        preprocessor_config=PreprocessorConfig(),
        tabular_dataset_kwargs={"name": "german"}),
    "diabetes_readmission": ExperimentConfig(
        splitter=RandomSplitter(test_size=0.25, val_size=0.01,
                                random_state=29746),
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

    "mimic_extract_los_3": ExperimentConfig(
        splitter=DomainSplitter(val_size=0.05,
                                id_test_size=0.2,
                                random_state=43456,
                                domain_split_varname="insurance",
                                domain_split_ood_values=[
                                    "Medicare", "Medicaid"]),
        grouper=Grouper({"gender": ['M'], }, drop=False),
        # We passthrough all non-static columns because we use
        # MIMIC-extract's default preprocessing/imputation and do not
        # wish to modify it for these features
        # (static features are not preprocessed by MIMIC-extract). See
        # tableshift.datasets.mimic_extract.preprocess_mimic_extract().
        preprocessor_config=PreprocessorConfig(
            passthrough_columns=[f for f in MIMIC_EXTRACT_SHARED_FEATURES.names
                                 if
                                 f not in MIMIC_EXTRACT_STATIC_FEATURES.names]),
        tabular_dataset_kwargs={"task": "los_3"}),

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
        splitter=RandomSplitter(test_size=0.5, val_size=0.25,
                                random_state=29746),
        # Race (non. hispanic white vs. all others; male vs. all others)
        grouper=Grouper({"RIDRETH3": ["3.0", ], "RIAGENDR": ["1.0", ]},
                        drop=False),
        preprocessor_config=PreprocessorConfig(
            passthrough_columns=["nhanes_year"],
            numeric_features="kbins"),
        tabular_dataset_kwargs={"nhanes_task": "cholesterol"}),

    "nhanes_lead": ExperimentConfig(
        splitter=RandomSplitter(test_size=0.5, val_size=0.25,
                                random_state=229446),
        # Race (non. hispanic white vs. all others; male vs. all others)
        grouper=Grouper({"RIDRETH3": ["3.0", ], "RIAGENDR": ["1.0", ]},
                        drop=False),
        preprocessor_config=PreprocessorConfig(
            passthrough_columns=["nhanes_year"],
            numeric_features="kbins"),
        tabular_dataset_kwargs={
            "nhanes_task": "lead",
            "years": [2007, 2009, 2011, 2013, 2015, 2017]}),

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

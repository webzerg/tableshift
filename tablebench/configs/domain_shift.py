from dataclasses import dataclass
from typing import Sequence, Optional, Any, Iterator

from tablebench.configs.experiment_defaults import DEFAULT_ID_TEST_SIZE, \
    DEFAULT_OOD_VAL_SIZE, DEFAULT_ID_VAL_SIZE, DEFAULT_RANDOM_STATE
from tablebench.core import Grouper, PreprocessorConfig, DomainSplitter
from tablebench.datasets import ACS_REGIONS, ACS_YEARS, \
    BRFSS_YEARS, CANDC_STATE_LIST, NHANES_YEARS, ANES_STATES, ANES_YEARS, \
    ANES_REGIONS, MIMIC_EXTRACT_SHARED_FEATURES, MIMIC_EXTRACT_STATIC_FEATURES
from tablebench.configs.experiment_configs import ExperimentConfig


def _to_nested(ary: Sequence[Any]) -> Sequence[Sequence[Any]]:
    """Create a nested tuple from a sequence.

    This reformats lists e.g. where each element in the list is the only desired
    out-of-domain value in an experiment.
    """
    return tuple([x] for x in ary)


@dataclass
class DomainShiftExperimentConfig:
    """Class to hold parameters for a domain shift experiment.

    This class defines a *set* of experiments, where the distribution split changes
    over experiments but all other factors (preprocessing, grouping, etc.) stay fixed.

    This class is used e.g. to identify which of a set of candidate domain splits has the
    biggest domain gap.
    """
    tabular_dataset_kwargs: dict
    domain_split_varname: str
    domain_split_ood_values: Sequence[Any]
    grouper: Optional[Grouper]
    preprocessor_config: PreprocessorConfig
    domain_split_id_values: Optional[Sequence[Any]] = None

    def as_experiment_config_iterator(
            self, val_size=DEFAULT_ID_VAL_SIZE,
            ood_val_size=DEFAULT_OOD_VAL_SIZE,
            id_test_size=DEFAULT_ID_TEST_SIZE,
            random_state=DEFAULT_RANDOM_STATE
    ) -> Iterator[ExperimentConfig]:
        for i, tgt in enumerate(self.domain_split_ood_values):
            if self.domain_split_id_values is not None:
                src = self.domain_split_id_values[i]
            else:
                src = None
            if not isinstance(tgt, tuple) and not isinstance(tgt, list):
                tgt = (tgt,)
            splitter = DomainSplitter(
                val_size=val_size,
                ood_val_size=ood_val_size,
                id_test_size=id_test_size,
                domain_split_varname=self.domain_split_varname,
                domain_split_ood_values=tgt,
                domain_split_id_values=src,
                random_state=random_state)
            yield ExperimentConfig(splitter=splitter, grouper=self.grouper,
                                   preprocessor_config=self.preprocessor_config,
                                   tabular_dataset_kwargs=self.tabular_dataset_kwargs)


# Set of fixed domain shift experiments.
domain_shift_experiment_configs = {
    "acsfoodstamps_region": DomainShiftExperimentConfig(
        tabular_dataset_kwargs={"name": "acsfoodstamps",
                                "acs_task": "acsfoodstamps"},
        domain_split_varname="DIVISION",
        domain_split_ood_values=_to_nested(ACS_REGIONS),
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

    "acsincome_region": DomainShiftExperimentConfig(
        tabular_dataset_kwargs={"name": "acsincome",
                                "acs_task": "acsincome"},
        domain_split_varname="DIVISION",
        domain_split_ood_values=_to_nested(ACS_REGIONS),
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

    "acspubcov_disability": DomainShiftExperimentConfig(
        tabular_dataset_kwargs={"name": "acspubcov",
                                "acs_task": "acspubcov",
                                "years": ACS_YEARS},
        domain_split_varname="DIS",
        domain_split_ood_values=_to_nested(['1.0']),
        grouper=Grouper({"RAC1P": [1, ], "SEX": [1, ]}, drop=False),
        preprocessor_config=PreprocessorConfig()),

    "acsunemployment_disability": DomainShiftExperimentConfig(
        tabular_dataset_kwargs={"name": "acsunemployment",
                                "acs_task": "acsunemployment",
                                "years": ACS_YEARS},
        domain_split_varname="DIS",
        domain_split_ood_values=_to_nested(['1.0']),
        grouper=Grouper({"RAC1P": [1, ], "SEX": [1, ]}, drop=False),
        preprocessor_config=PreprocessorConfig()),

    "acsunemployment_edlvl": DomainShiftExperimentConfig(
        tabular_dataset_kwargs={"name": "acsunemployment",
                                "acs_task": "acsunemployment",
                                "years": ACS_YEARS},
        domain_split_varname='SCHL',
        # No high school diploma vs. GED/diploma or higher.
        domain_split_ood_values=_to_nested(['01', '02', '03', '04',
                                            '05', '06', '07', '08',
                                            '09', '10', '11', '12',
                                            '13', '14', '15']),
        grouper=Grouper({"RAC1P": [1, ], "SEX": [1, ]}, drop=False),
        preprocessor_config=PreprocessorConfig()),

    "acsunemployment_mobility": DomainShiftExperimentConfig(
        tabular_dataset_kwargs={"name": "acsunemployment",
                                "acs_task": "acsunemployment",
                                "years": ACS_YEARS},
        domain_split_varname='MIG',
        domain_split_ood_values=_to_nested(['02', '03']),
        grouper=Grouper({"RAC1P": [1, ], "SEX": [1, ]}, drop=False),
        preprocessor_config=PreprocessorConfig()),

    "brfss_diabetes_race": DomainShiftExperimentConfig(
        tabular_dataset_kwargs={"name": "brfss_diabetes", "task": "diabetes",
                                "years": BRFSS_YEARS},
        domain_split_varname="PRACE1",
        # Train on white nonhispanic; test on all other race identities.
        domain_split_ood_values=[[2, 3, 4, 5, 6]],
        domain_split_id_values=_to_nested([1, ]),
        grouper=Grouper({"SEX": [1, ]}, drop=False),
        preprocessor_config=PreprocessorConfig(
            passthrough_columns=["IYEAR"]), ),

    "brfss_blood_pressure_income": DomainShiftExperimentConfig(
        tabular_dataset_kwargs={"name": "brfss_blood_pressure",
                                "task": "blood_pressure",
                                "years": BRFSS_YEARS},
        domain_split_varname="POVERTY",
        # Train on non-poverty observations; test (OOD) on poverty observations
        domain_split_ood_values=_to_nested([1, ]),
        domain_split_id_values=_to_nested([0, ]),
        grouper=Grouper({"SEX": [1, ]}, drop=False),
        preprocessor_config=PreprocessorConfig(
            passthrough_columns=["IYEAR"]), ),

    "brfss_blood_pressure_bmi": DomainShiftExperimentConfig(
        tabular_dataset_kwargs={"name": "brfss_blood_pressure",
                                "task": "blood_pressure",
                                "years": BRFSS_YEARS},
        domain_split_varname="BMI5CAT",
        # OOD values: [1 underweight, 2 normal weight], [3 overweight, 4 obese]
        domain_split_ood_values=[[1, 2], [3, 4]],
        grouper=Grouper({"SEX": [1, ]}, drop=False),
        preprocessor_config=PreprocessorConfig(
            passthrough_columns=["IYEAR"]), ),

    "candc_st": DomainShiftExperimentConfig(
        tabular_dataset_kwargs={"name": "communities_and_crime"},
        domain_split_varname="state",
        domain_split_ood_values=_to_nested(CANDC_STATE_LIST),
        grouper=Grouper({"Race": [1, ], "income_level_above_median": [1, ]},
                        drop=False),
        preprocessor_config=PreprocessorConfig(),
    ),

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
    "_debug": DomainShiftExperimentConfig(
        tabular_dataset_kwargs={"name": "german"},
        domain_split_varname="purpose",
        domain_split_ood_values=[["A44", "A410", "A45", "A46", "A48"]],
        grouper=Grouper({"sex": ['1', ], "age_geq_median": ['1', ]},
                        drop=False),
        preprocessor_config=PreprocessorConfig(),
    ),

    # Integer identifier corresponding to 21 distinct values, for example, physician referral,
    # emergency room, and transfer from a hospital,
    # https://downloads.hindawi.com/journals/bmri/2014/781670.pdf
    "diabetes_admsrc": DomainShiftExperimentConfig(
        tabular_dataset_kwargs={"name": "diabetes_readmission"},
        domain_split_varname='admission_source_id',
        domain_split_ood_values=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 14, 17,
                                 20, 22, 25],
        grouper=Grouper({"race": ["Caucasian", ], "gender": ["Male", ]},
                        drop=False),
        preprocessor_config=PreprocessorConfig(min_frequency=0.01),
    ),

    "heloc_burden": DomainShiftExperimentConfig(
        # HELOC task, where target domain are various upper-quantile
        # portions of the NetFractionRevolvingBurden feature.
        tabular_dataset_kwargs={"name": "heloc"},
        domain_split_varname='NetFractionRevolvingBurdenPercentile',
        domain_split_ood_values=[
            list(range(75, 101)), list(range(90, 101)),
            list(range(95, 101)), list(range(99, 101))
        ],
        grouper=None,
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

    "nhanes_cholesterol_race": DomainShiftExperimentConfig(
        tabular_dataset_kwargs={"name": "nhanes_cholesterol",
                                "nhanes_task": "cholesterol",
                                "years": NHANES_YEARS},
        domain_split_varname='RIDRETH_merged',
        domain_split_ood_values=[[1, 2, 4, 6, 7]],
        domain_split_id_values=[[3]],
        # Group by male vs. all others
        grouper=Grouper({"RIAGENDR": ["1.0", ]}, drop=False),
        preprocessor_config=PreprocessorConfig(
            passthrough_columns=["nhanes_year"],
            numeric_features="kbins")
    ),

    "nhanes_lead_poverty": DomainShiftExperimentConfig(
        tabular_dataset_kwargs={"name": "nhanes_lead",
                                "nhanes_task": "lead", "years": NHANES_YEARS},
        domain_split_varname='INDFMPIRBelowCutoff',
        domain_split_ood_values=[[1.]],
        # Race (non. hispanic white vs. all others; male vs. all others)
        grouper=Grouper({"RIDRETH_merged": [3, ], "RIAGENDR": ["1.0", ]},
                        drop=False),
        preprocessor_config=PreprocessorConfig(
            passthrough_columns=["nhanes_year"],
            numeric_features="kbins"),
    ),

    "physionet_set": DomainShiftExperimentConfig(
        tabular_dataset_kwargs={"name": "physionet"},
        domain_split_varname="set",
        domain_split_ood_values=_to_nested(["a", "b"]),
        grouper=Grouper({"Age": [x for x in range(40, 100)], "Gender": [1, ]},
                        drop=False),
        preprocessor_config=PreprocessorConfig(numeric_features="kbins",
                                               dropna=None)
    ),

    "anes_st": DomainShiftExperimentConfig(
        tabular_dataset_kwargs={"name": "anes", "years": [2020, ]},
        domain_split_varname="VCF0901b",
        domain_split_ood_values=_to_nested(ANES_STATES),
        grouper=Grouper({"VCF0104": ["1", ], "VCF0105a": ["1.0", ]},
                        drop=False),
        preprocessor_config=PreprocessorConfig(numeric_features="kbins",
                                               dropna=None)),

    "anes_region": DomainShiftExperimentConfig(
        tabular_dataset_kwargs={"name": "anes", "years": [2020, ]},
        domain_split_varname='VCF0112',
        domain_split_ood_values=_to_nested(ANES_REGIONS),
        grouper=Grouper({"VCF0104": ["1", ], "VCF0105a": ["1.0", ]},
                        drop=False),
        preprocessor_config=PreprocessorConfig(numeric_features="kbins",
                                               dropna=None)),

    # ANES: test on (2016) or (2020); train on all years prior.
    "anes_year": DomainShiftExperimentConfig(
        tabular_dataset_kwargs={"name": "anes"},
        domain_split_varname="VCF0004",
        domain_split_ood_values=[[ANES_YEARS[-2]], [ANES_YEARS[-1]]],
        domain_split_id_values=[ANES_YEARS[:-2], ANES_YEARS[:-1]],
        grouper=Grouper({"VCF0104": ["1", ], "VCF0105a": ["1.0", ]},
                        drop=False),
        preprocessor_config=PreprocessorConfig(numeric_features="kbins",
                                               dropna=None)),

    "mimic_extract_los_3_ins": DomainShiftExperimentConfig(
        tabular_dataset_kwargs={'task': 'los_3', 'name': 'mimic_extract_los_3'},
        domain_split_varname="insurance",
        domain_split_ood_values=_to_nested(
            ["Medicare", "Medicaid", "Government", "Self Pay"]),
        grouper=Grouper({"gender": ['M'], }, drop=False),
        # We passthrough all non-static columns because we use MIMIC-extract's default
        # preprocessing/imputation and do not wish to modify it for these features
        # (static features are not preprocessed by MIMIC-extract). See
        # tableshift.datasets.mimic_extract.preprocess_mimic_extract().
        preprocessor_config=PreprocessorConfig(
            passthrough_columns=[f for f in MIMIC_EXTRACT_SHARED_FEATURES.names
                                 if
                                 f not in MIMIC_EXTRACT_STATIC_FEATURES.names])),

    "mimic_extract_mort_hosp_ins": DomainShiftExperimentConfig(
        tabular_dataset_kwargs={'task': 'mort_hosp',
                                'name': 'mimic_extract_mort_hosp'},
        domain_split_varname="insurance",
        domain_split_ood_values=_to_nested(
            ["Medicare", "Medicaid", "Government", "Self Pay"]),
        grouper=Grouper({"gender": ['M'], }, drop=False),
        # We passthrough all non-static columns because we use MIMIC-extract's default
        # preprocessing/imputation and do not wish to modify it for these features
        # (static features are not preprocessed by MIMIC-extract). See
        # tableshift.datasets.mimic_extract.preprocess_mimic_extract().
        preprocessor_config=PreprocessorConfig(
            passthrough_columns=[f for f in MIMIC_EXTRACT_SHARED_FEATURES.names
                                 if
                                 f not in MIMIC_EXTRACT_STATIC_FEATURES.names])),
}

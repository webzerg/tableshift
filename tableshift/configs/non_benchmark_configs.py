"""
Experiment configs for the non-TableShift benchmark tasks.
"""

from tableshift.configs.experiment_config import ExperimentConfig

from tableshift.core import RandomSplitter, Grouper, PreprocessorConfig, \
    DomainSplitter, FixedSplitter

from tableshift.configs.experiment_defaults import DEFAULT_ID_TEST_SIZE, \
    DEFAULT_OOD_VAL_SIZE, DEFAULT_ID_VAL_SIZE, DEFAULT_RANDOM_STATE

NON_BENCHMARK_CONFIGS = {
    "adult": ExperimentConfig(
        splitter=FixedSplitter(val_size=0.25, random_state=29746),
        grouper=Grouper({"Race": ["White", ], "Sex": ["Male", ]}, drop=False),
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
        grouper=Grouper({"sex": ['1.0', ], "age_geq_median": ['1.0', ]},
                        drop=False),
        preprocessor_config=PreprocessorConfig(),
        tabular_dataset_kwargs={"name": "german"}),

    "german": ExperimentConfig(
        splitter=RandomSplitter(val_size=0.01, test_size=0.2, random_state=832),
        grouper=Grouper({"sex": ['1.0', ], "age_geq_median": ['1.0', ]},
                        drop=False),
        preprocessor_config=PreprocessorConfig(), tabular_dataset_kwargs={}),

    "metamimic_alcohol": ExperimentConfig(
        splitter=RandomSplitter(val_size=DEFAULT_ID_VAL_SIZE,
                                random_state=DEFAULT_RANDOM_STATE,
                                test_size=DEFAULT_ID_TEST_SIZE),
        grouper=None,
        preprocessor_config=PreprocessorConfig(
            numeric_features="kbins",
            passthrough_columns=["age"]
        ),
        tabular_dataset_kwargs={'name': 'metamimic_alcohol'}),

    'metamimic_anemia': ExperimentConfig(
        splitter=RandomSplitter(val_size=DEFAULT_ID_VAL_SIZE,
                                random_state=DEFAULT_RANDOM_STATE,
                                test_size=DEFAULT_ID_TEST_SIZE),
        grouper=None,
        preprocessor_config=PreprocessorConfig(
            numeric_features="kbins",
            passthrough_columns=["age"]
        ),
        tabular_dataset_kwargs={'name': 'metamimic_anemia'}),

    'metamimic_atrial': ExperimentConfig(
        splitter=RandomSplitter(val_size=DEFAULT_ID_VAL_SIZE,
                                random_state=DEFAULT_RANDOM_STATE,
                                test_size=DEFAULT_ID_TEST_SIZE),
        grouper=None,
        preprocessor_config=PreprocessorConfig(
            numeric_features="kbins",
            passthrough_columns=["age"]
        ),
        tabular_dataset_kwargs={'name': 'metamimic_atrial'}),

    'metamimic_diabetes': ExperimentConfig(
        splitter=RandomSplitter(val_size=DEFAULT_ID_VAL_SIZE,
                                random_state=DEFAULT_RANDOM_STATE,
                                test_size=DEFAULT_ID_TEST_SIZE),
        grouper=None,
        preprocessor_config=PreprocessorConfig(
            numeric_features="kbins",
            passthrough_columns=["age"]
        ),
        tabular_dataset_kwargs={'name': 'metamimic_diabetes'}),

    'metamimic_heart': ExperimentConfig(
        splitter=RandomSplitter(val_size=DEFAULT_ID_VAL_SIZE,
                                random_state=DEFAULT_RANDOM_STATE,
                                test_size=DEFAULT_ID_TEST_SIZE),
        grouper=None,
        preprocessor_config=PreprocessorConfig(
            numeric_features="kbins",
            passthrough_columns=["age"]
        ),
        tabular_dataset_kwargs={'name': 'metamimic_heart'}),

    'metamimic_hypertension': ExperimentConfig(
        splitter=RandomSplitter(val_size=DEFAULT_ID_VAL_SIZE,
                                random_state=DEFAULT_RANDOM_STATE,
                                test_size=DEFAULT_ID_TEST_SIZE),
        grouper=None,
        preprocessor_config=PreprocessorConfig(
            numeric_features="kbins",
            passthrough_columns=["age"]
        ),
        tabular_dataset_kwargs={'name': 'metamimic_hypertension'}),

    'metamimic_hypotension': ExperimentConfig(
        splitter=RandomSplitter(val_size=DEFAULT_ID_VAL_SIZE,
                                random_state=DEFAULT_RANDOM_STATE,
                                test_size=DEFAULT_ID_TEST_SIZE),
        grouper=None,
        preprocessor_config=PreprocessorConfig(
            numeric_features="kbins",
            passthrough_columns=["age"]
        ),
        tabular_dataset_kwargs={'name': 'metamimic_hypotension'}),

    'metamimic_ischematic': ExperimentConfig(
        splitter=RandomSplitter(val_size=DEFAULT_ID_VAL_SIZE,
                                random_state=DEFAULT_RANDOM_STATE,
                                test_size=DEFAULT_ID_TEST_SIZE),
        grouper=None,
        preprocessor_config=PreprocessorConfig(
            numeric_features="kbins",
            passthrough_columns=["age"]
        ),
        tabular_dataset_kwargs={'name': 'metamimic_ischematic'}),

    'metamimic_lipoid': ExperimentConfig(
        splitter=RandomSplitter(val_size=DEFAULT_ID_VAL_SIZE,
                                random_state=DEFAULT_RANDOM_STATE,
                                test_size=DEFAULT_ID_TEST_SIZE),
        grouper=None,
        preprocessor_config=PreprocessorConfig(
            numeric_features="kbins",
            passthrough_columns=["age"]
        ),
        tabular_dataset_kwargs={'name': 'metamimic_lipoid'}),

    'metamimic_overweight': ExperimentConfig(
        splitter=RandomSplitter(val_size=DEFAULT_ID_VAL_SIZE,
                                random_state=DEFAULT_RANDOM_STATE,
                                test_size=DEFAULT_ID_TEST_SIZE),
        grouper=None,
        preprocessor_config=PreprocessorConfig(
            numeric_features="kbins",
            passthrough_columns=["age"]
        ),
        tabular_dataset_kwargs={'name': 'metamimic_overweight'}),

    'metamimic_purpura': ExperimentConfig(
        splitter=RandomSplitter(val_size=DEFAULT_ID_VAL_SIZE,
                                random_state=DEFAULT_RANDOM_STATE,
                                test_size=DEFAULT_ID_TEST_SIZE),
        grouper=None,
        preprocessor_config=PreprocessorConfig(
            numeric_features="kbins",
            passthrough_columns=["age"]
        ),
        tabular_dataset_kwargs={'name': 'metamimic_purpura'}),

    'metamimic_respiratory': ExperimentConfig(
        splitter=RandomSplitter(val_size=DEFAULT_ID_VAL_SIZE,
                                random_state=DEFAULT_RANDOM_STATE,
                                test_size=DEFAULT_ID_TEST_SIZE),
        grouper=None,
        preprocessor_config=PreprocessorConfig(
            numeric_features="kbins",
            passthrough_columns=["age"]
        ),
        tabular_dataset_kwargs={'name': 'metamimic_respiratory'}),

    "mooc": ExperimentConfig(
        splitter=DomainSplitter(val_size=DEFAULT_ID_VAL_SIZE,
                                ood_val_size=DEFAULT_OOD_VAL_SIZE,
                                random_state=DEFAULT_RANDOM_STATE,
                                id_test_size=DEFAULT_ID_TEST_SIZE,
                                domain_split_varname="course_id",
                                domain_split_ood_values=[
                                    "HarvardX/CB22x/2013_Spring"]),
        grouper=Grouper({"gender": ["m", ],
                         "LoE_DI": ["Bachelor's", "Master's", "Doctorate"]},
                        drop=False),
        preprocessor_config=PreprocessorConfig(), tabular_dataset_kwargs={}),
}

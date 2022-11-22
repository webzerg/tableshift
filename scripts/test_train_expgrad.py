from tablebench.core import TabularDataset, TabularDatasetConfig, \
    DomainSplitter, Grouper, PreprocessorConfig

from fairlearn.reductions import ErrorRateParity
import xgboost as xgb

from tablebench.datasets.experiment_configs import ExperimentConfig
from tablebench.models import get_estimator

experiment = "german"
domain_split_varname = "purpose"
expt_config = ExperimentConfig(
    splitter=DomainSplitter(val_size=0.01,
                            id_test_size=0.2,
                            random_state=43406,
                            domain_split_varname=domain_split_varname,
                            # OOD values: car(new), car(used), furniture/equipment
                            domain_split_ood_values=["A40", "A41", "A42", "A43"],
                            drop_domain_split_col=False),
    grouper=Grouper({"sex": ['1', ], "age_geq_median": ['1', ]},
                    drop=False),
    preprocessor_config=PreprocessorConfig(),
    tabular_dataset_kwargs={"name": experiment})

dataset_config = TabularDatasetConfig()

dset = TabularDataset(config=dataset_config,
                      splitter=expt_config.splitter,
                      grouper=expt_config.grouper,
                      preprocessor_config=expt_config.preprocessor_config,
                      **expt_config.tabular_dataset_kwargs)

X_tr, y_tr, _ = dset.get_pandas(split="train")

base_estimator = xgb.XGBClassifier()
constraint = ErrorRateParity()
estimator = get_estimator(
    "expgrad",
    estimator=base_estimator,
    constraints=constraint,
    domain_feature_colname=[domain_split_varname])

estimator.fit(X_tr, y_tr)

for split in ("id_test", "ood_test"):

    X_te, _, _ = dset.get_pandas(split=split)

    y_hat_te = estimator.predict(X_te)
    metrics = dset.evaluate_predictions(y_hat_te, split=split)
    print(f"metrics on split {split}:")
    for k, v in metrics.items():
        print(f"\t{k:<40}:{v:.3f}")

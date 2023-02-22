from tableshift.core import TabularDataset, TabularDatasetConfig, \
    DomainSplitter, Grouper, PreprocessorConfig

from tableshift.configs.experiment_configs import ExperimentConfig
from tableshift.models.utils import get_estimator

experiment = "mooc"
expt_config = ExperimentConfig(
    splitter=DomainSplitter(val_size=0.01,
                            id_test_size=0.2,
                            ood_val_size=0.5,
                            random_state=43406,
                            domain_split_varname="course_id",
                            domain_split_ood_values=[
                                "HarvardX/CB22x/2013_Spring"]),
    grouper=Grouper({"gender": ["m", ],
                     "LoE_DI": ["Bachelor's", "Master's", "Doctorate"]},
                    drop=False),
    preprocessor_config=PreprocessorConfig(), tabular_dataset_kwargs={})

dataset_config = TabularDatasetConfig()
dset = TabularDataset(experiment,
                      config=dataset_config,
                      splitter=expt_config.splitter,
                      grouper=expt_config.grouper,
                      preprocessor_config=expt_config.preprocessor_config,
                      **expt_config.tabular_dataset_kwargs)

X_tr, y_tr, _, _ = dset.get_pandas(split="train")
X_ood_tr, y_ood_tr, _, _ = dset.get_pandas(split="ood_validation")
estimator = get_estimator("wcs", C_domain=1., C_discrim=1.)
estimator.fit(X_tr, y_tr, X_ood_tr)

for split in ("id_test", "ood_test"):

    X_te, _, _, _ = dset.get_pandas(split=split)

    y_hat_te = estimator.predict(X_te)
    metrics = dset.evaluate_predictions(y_hat_te, split=split)
    print(f"metrics on split {split}:")
    for k, v in metrics.items():
        print(f"\t{k:<40}:{v:.3f}")

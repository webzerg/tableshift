from tablebench.core import TabularDataset, TabularDatasetConfig

from tablebench.datasets.experiment_configs import EXPERIMENT_CONFIGS
from tablebench.models.utils import get_estimator
from tablebench.models.config import get_model_config

expt_config = EXPERIMENT_CONFIGS["_debug"]

dataset_config = TabularDatasetConfig()

dset = TabularDataset(config=dataset_config,
                      splitter=expt_config.splitter,
                      grouper=expt_config.grouper,
                      preprocessor_config=expt_config.preprocessor_config,
                      **expt_config.tabular_dataset_kwargs)

X_tr, y_tr, _, d_tr = dset.get_pandas(split="train")

config = get_model_config("expgrad", dset)
estimator = get_estimator("expgrad", **config)

estimator.fit(X_tr, y_tr, d=d_tr)

for split in ("id_test", "ood_test"):

    X_te, _, _, _ = dset.get_pandas(split=split)

    y_hat_te = estimator.predict(X_te)
    metrics = dset.evaluate_predictions(y_hat_te, split=split)
    print(f"metrics on split {split}:")
    for k, v in metrics.items():
        print(f"\t{k:<40}:{v:.3f}")

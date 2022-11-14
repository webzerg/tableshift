import argparse
from tablebench.core import DomainSplitter, TabularDataset, \
    TabularDatasetConfig

from tablebench.datasets.experiment_configs import EXPERIMENT_CONFIGS
from tablebench.models import get_estimator


def main(experiment, cache_dir, model):
    if experiment not in EXPERIMENT_CONFIGS:
        raise NotImplementedError(f"{experiment} is not implemented.")

    expt_config = EXPERIMENT_CONFIGS[experiment]
    dataset_config = TabularDatasetConfig(cache_dir=cache_dir)
    dset = TabularDataset(experiment,
                          config=dataset_config,
                          splitter=expt_config.splitter,
                          grouper=expt_config.grouper,
                          preprocessor_config=expt_config.preprocessor_config,
                          **expt_config.tabular_dataset_kwargs)
    X_tr, y_tr, G_tr = dset.get_pandas(split="train")
    estimator = get_estimator(model)

    print(f"fitting estimator of type {type(estimator)}")
    estimator.fit(X_tr, y_tr)
    print("fitting estimator complete.")

    splits = ("test",) if not isinstance(
        expt_config.splitter, DomainSplitter) \
        else ("id_test", "ood_test")

    for split in splits:

        X_te, _, _ = dset.get_pandas(split=split)

        y_hat_te = estimator.predict(X_te)
        metrics = dset.evaluate_predictions(y_hat_te, split=split)
        print(f"metrics on split {split}:")
        for k, v in metrics.items():
            print(f"\t{k:<40}:{v:.3f}")
    return


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment", choices=list(EXPERIMENT_CONFIGS.keys()),
                        default="brfss_diabetes_st")
    parser.add_argument("--cache_dir", default="tmp",
                        help="Directory to cache raw data files to.")
    parser.add_argument("--model", default="histgbm",
                        help="model to use.")
    args = parser.parse_args()
    main(**vars(args))

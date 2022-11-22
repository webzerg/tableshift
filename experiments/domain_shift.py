import argparse
from datetime import datetime

import pandas as pd
from tablebench.models import get_estimator, PYTORCH_MODELS, SKLEARN_MODELS, \
    training

from tablebench.configs.domain_shift import domain_shift_experiment_configs
from tablebench.core import DomainSplitter, TabularDataset, \
    TabularDatasetConfig


def main(experiment, cache_dir, device: str):
    iterates = []

    expt_config = domain_shift_experiment_configs[experiment]
    dataset_config = TabularDatasetConfig(cache_dir=cache_dir)
    for i, tgt in enumerate(expt_config.domain_split_ood_values):

        if expt_config.domain_split_id_values is not None:
            src = expt_config.domain_split_id_values[i]
        else:
            src = None

        splitter = DomainSplitter(
            val_size=0.01,
            id_test_size=1 / 5.,
            domain_split_varname=expt_config.domain_split_varname,
            domain_split_ood_values=[tgt],
            domain_split_id_values=src,
            random_state=19542)

        try:
            dset = TabularDataset(
                **expt_config.tabular_dataset_kwargs,
                config=dataset_config,
                splitter=splitter,
                grouper=expt_config.grouper,
                preprocessor_config=expt_config.preprocessor_config)
        except ValueError as ve:
            # Case: split is too small.
            print(f"[WARNING] error initializing dataset for expt {experiment} "
                  f"with {expt_config.domain_split_varname} == {tgt}: {ve}")
            continue

        eval_splits = ("test",) if not isinstance(
            expt_config.splitter, DomainSplitter) else ("id_test", "ood_test")

        for model in list(PYTORCH_MODELS) + list(SKLEARN_MODELS):
            estimator = get_estimator(model, d_out=dset.X_shape[1])

            print(f"fitting estimator of type {type(estimator)} with "
                  f"target {expt_config.domain_split_varname} = {tgt}, "
                  f"src {expt_config.domain_split_varname} = {src}")

            if model in SKLEARN_MODELS:
                estimator = training.train_sklearn(estimator, dset, eval_splits)

            else:
                estimator = training.train_pytorch(model, dset, device,
                                                   eval_splits)

            metrics = {"estimator": str(type(estimator)),
                       "task": expt_config.tabular_dataset_kwargs[
                           "name"],
                       "domain_split_varname": expt_config.domain_split_varname,
                       "domain_split_ood_values": tgt,
                       }
            for split in ("id_test", "ood_test"):
                try:
                    X_te, _, _ = dset.get_pandas(split=split)

                    y_hat_te = estimator.predict(X_te)
                    split_metrics = dset.evaluate_predictions(y_hat_te,
                                                              split=split)
                    metrics.update(split_metrics)
                except Exception as e:
                    print(f"exception evaluating split {split} with "
                          f"{expt_config.domain_split_varname}=={tgt}: "
                          f"{e}; skipping")
                    continue
            iterates.append(metrics)

        results = pd.DataFrame(iterates)
        results.to_csv(
            f"results-{experiment}-{str(datetime.now()).replace(' ', '')}.csv")
        return

    if __name__ == "__main__":
        parser = argparse.ArgumentParser()
        parser.add_argument("--cache_dir", default="tmp",
                            help="Directory to cache raw data files to.")
        parser.add_argument("--device", default="cpu")
        parser.add_argument("--experiment",
                            choices=list(
                                domain_shift_experiment_configs.keys()),
                            default="brfss_diabetes_st")
        args = parser.parse_args()
        main(**vars(args))

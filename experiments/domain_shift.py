import argparse
from datetime import datetime

import pandas as pd
from tablebench.models import get_estimator, get_pytorch_model_config, \
    is_pytorch_model_name, PYTORCH_MODEL_CLS, SKLEARN_MODEL_CLS
from tablebench.models.training import train
from tablebench.models.utils import get_predictions_and_labels

from tablebench.configs.domain_shift import domain_shift_experiment_configs
from tablebench.core import DomainSplitter, TabularDataset, \
    TabularDatasetConfig


def main(experiment, cache_dir, device: str):

    # List of dictionaries containing metrics and metadata for each
    # experimental iterate.
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

        for model in PYTORCH_MODEL_CLS.keys():
            # for model in list(PYTORCH_MODEL_CLS) + list(SKLEARN_MODELS):

            config = get_pytorch_model_config(
                model, dset) if is_pytorch_model_name(model) else {}

            estimator = get_estimator(model, **config)

            print(f"[INFO] fitting estimator of type {type(estimator)} with "
                  f"target {expt_config.domain_split_varname} = {tgt}, "
                  f"src {expt_config.domain_split_varname} = {src}")

            estimator = train(estimator, dset, device=device)

            # Initialize the metrics dict with some experiment metadata.
            metrics = {"estimator": str(type(estimator)),
                       "task": expt_config.tabular_dataset_kwargs[
                           "name"],
                       "domain_split_varname": expt_config.domain_split_varname,
                       "domain_split_ood_values": tgt}

            for split in dset.eval_split_names:
                try:
                    if model in PYTORCH_MODEL_CLS:
                        loader = dset.get_dataloader(split)
                        y_hat, _ = get_predictions_and_labels(estimator, loader)

                    else:
                        X_te, _, _ = dset.get_pandas(split=split)
                        y_hat = estimator.predict(X_te)

                    split_metrics = dset.evaluate_predictions(y_hat, split)
                    metrics.update(split_metrics)
                except Exception as e:
                    print(f"exception evaluating split {split} with "
                          f"{expt_config.domain_split_varname}=={tgt}: "
                          f"{e}; skipping")
                    continue
            iterates.append(metrics)

    results = pd.DataFrame(iterates)
    fp = f"results-{experiment}-{str(datetime.now()).replace(' ', '')}.csv"
    print(f"[INFO] writing results to {fp}")
    results.to_csv(fp)
    return


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache_dir", default="tmp",
                        help="Directory to cache raw data files to.")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--experiment",
                        choices=list(
                            domain_shift_experiment_configs.keys()),
                        default="mooc_course")
    args = parser.parse_args()
    main(**vars(args))

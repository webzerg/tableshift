"""
Run training/tuning of a single model.
"""
import argparse
from datetime import datetime

import pandas as pd
from tablebench.models import PYTORCH_MODEL_CLS, SKLEARN_MODEL_CLS
from tablebench.models.tuning import TuneConfig, run_tuning_experiment
from tablebench.models.utils import get_predictions_and_labels

from tablebench.configs.domain_shift import domain_shift_experiment_configs
from tablebench.core import DomainSplitter, TabularDataset, \
    TabularDatasetConfig


def main(experiment, cache_dir, device: str, debug: bool, no_tune: bool,
         num_samples: int, tune_metric_name: str = "metric",
         tune_metric_higher_is_better: bool = True):
    if debug:
        print("[INFO] running in debug mode.")
        del experiment
        experiment = "_debug"

    # List of dictionaries containing metrics and metadata for each
    # experimental iterate.
    iterates = []

    expt_config = domain_shift_experiment_configs[experiment]
    dataset_config = TabularDatasetConfig(cache_dir=cache_dir)

    tune_config = TuneConfig(
        num_samples=num_samples,
        tune_metric_name=tune_metric_name,
        tune_metric_higher_is_better=tune_metric_higher_is_better
    ) if not no_tune else None

    ood_values = expt_config.domain_split_ood_values
    if debug:
        # Just test the first ood split values.
        ood_values = [ood_values[0]]

    for i, tgt in enumerate(ood_values):

        if expt_config.domain_split_id_values is not None:
            src = expt_config.domain_split_id_values[i]
        else:
            src = None
        if not isinstance(tgt, tuple) and not isinstance(tgt, list):
            tgt = (tgt,)
        splitter = DomainSplitter(
            val_size=0.01,
            ood_val_size=0.1,
            id_test_size=1 / 5.,
            domain_split_varname=expt_config.domain_split_varname,
            domain_split_ood_values=tgt,
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

        for model in list(PYTORCH_MODEL_CLS) + list(SKLEARN_MODEL_CLS):
            results = run_tuning_experiment(model=model, dset=dset,
                                            device=device,
                                            tune_config=tune_config)

            best_result = results.get_best_result(
                tune_config.tune_metric_name, tune_config.mode)

            print("Best trial config: {}".format(best_result.config))
            best_metric = best_result.metrics[tune_config.tune_metric_name]
            print("Best trial final {}: {}".format(tune_config.tune_metric_name,
                                                   best_metric))

            # Initialize the metrics dict with some experiment metadata.
            metrics = {"estimator": model,
                       "task": expt_config.tabular_dataset_kwargs["name"],
                       "domain_split_varname": expt_config.domain_split_varname,
                       "domain_split_ood_values": tgt,
                       "tune_metric_name": tune_config.tune_metric_name,
                       "tune_metric_best_value": best_metric}

            # TODO(jpgard): get the best model and evaluate it on all of the
            #  splits; or, evaluate the best model inside
            #  run_tuning_experiment() or another new function?
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
    parser.add_argument("--debug", action="store_true", default=False,
                        help="Whether to run in debug mode. If True, various "
                             "truncations/simplifications are performed to "
                             "speed up experiment.")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--experiment",
                        choices=list(domain_shift_experiment_configs.keys()),
                        default="mooc_course")
    parser.add_argument("--no_tune", action="store_true", default=False,
                        help="If set, suppresses hyperparameter tuning of the "
                             "model (for faster testing).")
    parser.add_argument("--num_samples", type=int, default=1,
                        help="Number of hparam samples to take in tuning "
                             "sweep.")
    args = parser.parse_args()
    main(**vars(args))

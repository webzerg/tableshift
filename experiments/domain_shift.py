import argparse

import pandas as pd

from tablebench.configs.domain_shift import domain_shift_experiment_configs
from tablebench.core import TabularDataset, TabularDatasetConfig, DomainSplitter
from tablebench.models.ray_utils import TuneConfig, run_ray_tune_experiment


def main(experiment: str, cache_dir: str,
         debug: bool,
         no_tune: bool,
         num_samples: int,
         tune_metric_name: str = "validation_accuracy",
         tune_metric_higher_is_better: bool = True,
         max_concurrent_trials=2,
         num_workers=1,
         early_stop=True):
    models = ("mlp", "resnet", "ft_transformer", "group_dro", "xgb", "lightgbm")

    if debug:
        print("[INFO] running in debug mode.")
        experiment = "_debug"
        num_samples = 1

    # List of dictionaries containing metrics and metadata for each
    # experimental iterate.
    iterates = []

    expt_config = domain_shift_experiment_configs[experiment]
    dataset_config = TabularDatasetConfig(cache_dir=cache_dir)

    ood_values = expt_config.domain_split_ood_values
    if debug:
        # Just test the first ood split values.
        ood_values = [ood_values[0]]

    tabular_dataset_kwargs = expt_config.tabular_dataset_kwargs
    if "name" not in tabular_dataset_kwargs:
        tabular_dataset_kwargs["name"] = experiment

    tune_config = TuneConfig(
        early_stop=early_stop,
        max_concurrent_trials=max_concurrent_trials,
        num_workers=num_workers,
        num_samples=num_samples,
        tune_metric_name=tune_metric_name,
        tune_metric_higher_is_better=tune_metric_higher_is_better,
    ) if not no_tune else None

    for i, tgt in enumerate(ood_values):

        if expt_config.domain_split_id_values is not None:
            src = expt_config.domain_split_id_values[i]
        else:
            src = None

        if not isinstance(tgt, tuple) and not isinstance(tgt, list):
            tgt = (tgt,)
        splitter = DomainSplitter(
            val_size=0.1,
            ood_val_size=0.1,
            id_test_size=0.1,
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

        for model_name in models:
            results = run_ray_tune_experiment(dset=dset, model_name=model_name,
                                              tune_config=tune_config)

            df = results.get_dataframe()
            df["estimator"] = model_name
            df["task"] = expt_config.tabular_dataset_kwargs["name"],
            df["domain_split_varname"] = expt_config.domain_split_varname
            df["domain_split_ood_values"] = str(tgt)

            print(df)
            best_result = results.get_best_result()

            print("Best trial config: {}".format(best_result.config))
            print("Best trial result: {}".format(best_result))

            df.to_csv(f"tune_results_{experiment}_{model_name}.csv",
                      index=False)
            iterates.append(df)

    fp = f"tune_results_{experiment}.csv"
    print(f"[INFO] writing results to {fp}")
    pd.concat(iterates).to_csv(fp, index=False)
    return


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache_dir", default="tmp",
                        help="Directory to cache raw data files to.")
    parser.add_argument("--debug", action="store_true", default=False,
                        help="Whether to run in debug mode. If True, various "
                             "truncations/simplifications are performed to "
                             "speed up experiment.")
    parser.add_argument("--experiment", default="adult",
                        help="Experiment to run. Overridden when debug=True.")
    parser.add_argument("--num_samples", type=int, default=1,
                        help="Number of hparam samples to take in tuning "
                             "sweep.")
    parser.add_argument("--no_tune", action="store_true", default=False,
                        help="If set, suppresses hyperparameter tuning of the "
                             "model (for faster testing).")
    args = parser.parse_args()
    main(**vars(args))

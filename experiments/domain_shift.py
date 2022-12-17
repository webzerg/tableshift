import argparse
from typing import Union

import pandas as pd

from tablebench.configs.domain_shift import domain_shift_experiment_configs
from tablebench.core import TabularDataset, TabularDatasetConfig, DomainSplitter, CachedDataset
from tablebench.models.ray_utils import TuneConfig, run_ray_tune_experiment, fetch_postprocessed_results_df
from tablebench.datasets.experiment_configs import ExperimentConfig
from tablebench.core.utils import make_uid


def get_dataset(expt_config: ExperimentConfig, dataset_config: TabularDatasetConfig,
                use_cached: bool) -> Union[TabularDataset, CachedDataset]:
    name = expt_config.tabular_dataset_kwargs["name"]
    if use_cached:
        uid = make_uid(name, expt_config.splitter)
        print(f"[INFO] using cached dataset for uid {uid} at {dataset_config.cache_dir}")
        dset = CachedDataset(cache_dir=dataset_config.cache_dir, name=name, uid=uid)
        return dset
    else:
        try:
            dset = TabularDataset(
                **expt_config.tabular_dataset_kwargs,
                config=dataset_config,
                splitter=expt_config.splitter,
                grouper=expt_config.grouper,
                preprocessor_config=expt_config.preprocessor_config)
            return dset
        except ValueError as ve:
            # Case: split is too small.
            print(f"[WARNING] error initializing dataset for expt {name} "
                  f"with {expt_config.splitter.domain_split_varname} == {tgt}: {ve}")
            return None


def main(experiment: str, cache_dir: str,
         debug: bool,
         no_tune: bool,
         num_samples: int,
         tune_metric_name: str = "validation_accuracy",
         tune_metric_higher_is_better: bool = True,
         max_concurrent_trials=2,
         num_workers=1,
         early_stop=True,
         use_cached: bool = False,
         time_budget_hrs: float = None):
    # Use baseline models only.
    models = (
        # "mlp",
        # "resnet",
        # "ft_transformer",
        # # "group_dro",
        "xgb",
        # "lightgbm"
    )

    if debug:
        print("[INFO] running in debug mode.")
        experiment = "_debug"
        num_samples = 1

    # List of dictionaries containing metrics and metadata for each
    # experimental iterate.
    iterates = []

    domain_shift_expt_config = domain_shift_experiment_configs[experiment]
    dataset_config = TabularDatasetConfig(cache_dir=cache_dir)

    ood_values = domain_shift_expt_config.domain_split_ood_values
    if debug:
        # Just test the first ood split values.
        ood_values = [ood_values[0]]

    tabular_dataset_kwargs = domain_shift_expt_config.tabular_dataset_kwargs
    if "name" not in tabular_dataset_kwargs:
        tabular_dataset_kwargs["name"] = experiment

    tune_config = TuneConfig(
        early_stop=early_stop,
        max_concurrent_trials=max_concurrent_trials,
        num_workers=num_workers,
        num_samples=num_samples,
        tune_metric_name=tune_metric_name,
        tune_metric_higher_is_better=tune_metric_higher_is_better,
        time_budget_hrs=time_budget_hrs,
    ) if not no_tune else None

    for expt_config in domain_shift_expt_config.as_experiment_config_iterator():

        assert isinstance(expt_config.splitter, DomainSplitter)
        if expt_config.splitter.domain_split_id_values is not None:
            src = expt_config.splitter.domain_split_id_values[i]
        else:
            src = None

        tgt = expt_config.splitter.domain_split_ood_values
        if not isinstance(tgt, tuple) and not isinstance(tgt, list):
            tgt = (tgt,)

        dset = get_dataset(expt_config=expt_config,
                           dataset_config=dataset_config, use_cached=use_cached)

        if use_cached and not dset.is_cached():
            print(f"[INFO] skipping dataset {dset.name}; not cached. This may not be a problem and be due to an "
                  f"invalid domain split (i.e. a domain split with only one target label).")
            continue

        for model_name in models:
            results = run_ray_tune_experiment(dset=dset, model_name=model_name,
                                              tune_config=tune_config, debug=debug)

            df = fetch_postprocessed_results_df(results)
            df["estimator"] = model_name
            df["task"] = expt_config.tabular_dataset_kwargs["name"],
            df["domain_split_varname"] = expt_config.splitter.domain_split_varname
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
    parser.add_argument("--time_budget_hrs", type=float, default=None,
                        help="Time budget for each model tuning run, in hours. Fractional hours are ok."
                             "If this is set, num_samples has no effect.")
    parser.add_argument("--no_tune", action="store_true", default=False,
                        help="If set, suppresses hyperparameter tuning of the "
                             "model (for faster testing).")
    parser.add_argument("--use_cached", action="store_true", default=False,
                        help="Whether to use cached data. If set to True,"
                             "and cached data does not exist, the job will fail.")
    args = parser.parse_args()
    main(**vars(args))

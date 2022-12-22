import argparse

from tablebench.core import CachedDataset
from tablebench.models.ray_utils import RayExperimentConfig, run_ray_tune_experiment, \
    accuracy_metric_name_and_mode_for_model, \
    fetch_postprocessed_results_df
from tablebench.datasets.experiment_configs import EXPERIMENT_CONFIGS
from tablebench.core import TabularDataset, TabularDatasetConfig



def main(experiment: str, uid: str, model_name: str, cache_dir: str,
         debug: bool,
         no_tune: bool, num_samples: int, search_alg: str,
         use_cached: bool,
         max_concurrent_trials=2,
         num_workers=1,
         early_stop=True):
    if use_cached:
        print(f"[DEBUG] loading cached data from {cache_dir}")
        dset = CachedDataset(cache_dir=cache_dir, name=experiment, uid=uid)
    else:
        expt_config = EXPERIMENT_CONFIGS[experiment]

        dataset_config = TabularDatasetConfig(cache_dir=cache_dir)
        tabular_dataset_kwargs = expt_config.tabular_dataset_kwargs
        if "name" not in tabular_dataset_kwargs:
            tabular_dataset_kwargs["name"] = experiment

        dset = TabularDataset(config=dataset_config,
                              splitter=expt_config.splitter,
                              grouper=expt_config.grouper,
                              preprocessor_config=expt_config.preprocessor_config,
                              **tabular_dataset_kwargs)

    metric_name, mode = accuracy_metric_name_and_mode_for_model(model_name)

    tune_config = RayExperimentConfig(
        early_stop=early_stop,
        max_concurrent_trials=max_concurrent_trials,
        num_workers=num_workers,
        num_samples=num_samples,
        tune_metric_name=metric_name,
        search_alg=search_alg,
        mode=mode) if not no_tune else None

    results = run_ray_tune_experiment(dset=dset, model_name=model_name,
                                      tune_config=tune_config, debug=debug)

    results_df = results.get_dataframe()
    print(results_df)
    results_df.to_csv(f"tune_results_{experiment}_{model_name}.csv",
                      index=False)
    print(results.get_best_result())
    # call fetc_posprocessed() just to match the full training loop
    df = fetch_postprocessed_results_df(results)
    print(df)
    return


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache_dir", default="tmp",
                        help="Directory to cache raw data files to.")
    parser.add_argument("--debug", action="store_true", default=False,
                        help="Whether to run in debug mode. If True, various "
                             "truncations/simplifications are performed to "
                             "speed up experiment.")
    parser.add_argument("--experiment", default="diabetes_readmission",
                        help="Experiment to run. Overridden when debug=True.")
    parser.add_argument("--model_name", default="mlp")
    parser.add_argument("--num_samples", type=int, default=1,
                        help="Number of hparam samples to take in tuning "
                             "sweep.")
    parser.add_argument("--no_tune", action="store_true", default=False,
                        help="If set, suppresses hyperparameter tuning of the "
                             "model (for faster testing).")
    parser.add_argument("--search_alg", default="hyperopt", choices=["hyperopt", "random"],
                        help="Ray search alg to use for hyperparameter tuning.")
    parser.add_argument("--uid",
                        default="diabetes_readmissiondomain_split_varname_admission_type_iddomain_split_ood_value_1",
                        help="UID for experiment to run. Overridden when debug=True.")
    parser.add_argument("--use_cached", default=False, action="store_true",
                        help="whether to use cached data.")
    args = parser.parse_args()
    main(**vars(args))

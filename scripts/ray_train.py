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
         gpu_per_worker: float = 1.0,
         scheduler: str = None):
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
        max_concurrent_trials=max_concurrent_trials,
        num_workers=num_workers,
        num_samples=num_samples,
        tune_metric_name=metric_name,
        search_alg=search_alg,
        scheduler=scheduler,
        gpu_per_worker=gpu_per_worker,
        mode=mode) if not no_tune else None

    results = run_ray_tune_experiment(dset=dset, model_name=model_name,
                                      tune_config=tune_config, debug=debug)

    results_df = results.get_dataframe()
    print(results_df)
    fp = f"tune_results_{uid}_{model_name}_{search_alg}.csv"
    print(f"[INFO] writing completed results to {fp}")
    results_df.to_csv(fp, index=False)

    # call fetch_postprocessed() just to match the full training loop
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
    parser.add_argument("--gpu_per_worker", default=1.0, type=float,
                        help="GPUs per worker. Use fractional values < 1. "
                             "(e.g. --gpu_per_worker=0.5) in order"
                             "to allow multiple workers to share GPU.")
    parser.add_argument("--model_name", default="mlp")
    parser.add_argument("--num_samples", type=int, default=1,
                        help="Number of hparam samples to take in tuning "
                             "sweep.")
    parser.add_argument("--no_tune", action="store_true", default=False,
                        help="If set, suppresses hyperparameter tuning of the "
                             "model (for faster testing).")
    parser.add_argument("--scheduler", choices=(None, "asha", "median"),
                        default=None,
                        help="Scheduler to use for hyperparameter optimization."
                             "See https://docs.ray.io/en/latest/tune/api_docs/schedulers.html .")
    parser.add_argument("--search_alg", default="hyperopt", choices=["hyperopt", "random"],
                        help="Ray search alg to use for hyperparameter tuning.")
    parser.add_argument("--uid",
                        default="diabetes_readmissiondomain_split_varname_admission_type_iddomain_split_ood_value_1",
                        help="UID for experiment to run. Overridden when debug=True.")
    parser.add_argument("--use_cached", default=False, action="store_true",
                        help="whether to use cached data.")
    args = parser.parse_args()
    main(**vars(args))

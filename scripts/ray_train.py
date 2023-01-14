import argparse
import os
from typing import Optional

from tablebench.core import CachedDataset
from tablebench.models.ray_utils import RayExperimentConfig, \
    run_ray_tune_experiment, \
    accuracy_metric_name_and_mode_for_model, \
    fetch_postprocessed_results_df
from tablebench.configs.experiment_configs import EXPERIMENT_CONFIGS
from tablebench.core import TabularDataset, TabularDatasetConfig
from tablebench.configs.ray_configs import get_default_ray_tmp_dir, \
    get_default_ray_local_dir


def main(experiment: str, uid: str, model_name: str, cache_dir: str,
         ray_tmp_dir: str,
         ray_local_dir: str,
         debug: bool,
         no_tune: bool, num_samples: int, search_alg: str,
         use_cached: bool,
         max_concurrent_trials=2,
         num_workers=1,
         gpu_per_worker: float = 1.0,
         cpu_per_worker: int = 1,
         scheduler: str = None):
    if not ray_tmp_dir:
        ray_tmp_dir = get_default_ray_tmp_dir()
    if not ray_local_dir:
        ray_local_dir = get_default_ray_local_dir()

    if debug:
        print("[INFO] running in debug mode.")
        experiment = "_debug"

    if use_cached:
        print(f"[DEBUG] loading cached data from {cache_dir}")
        assert uid is not None, "uid is required to use a cached dataset."
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
        ray_tmp_dir=ray_tmp_dir,
        ray_local_dir=ray_local_dir,
        num_workers=num_workers,
        num_samples=num_samples,
        tune_metric_name=metric_name,
        search_alg=search_alg,
        scheduler=scheduler,
        gpu_per_worker=gpu_per_worker,
        cpu_per_worker=cpu_per_worker,
        mode=mode) if not no_tune else None

    results = run_ray_tune_experiment(dset=dset, model_name=model_name,
                                      tune_config=tune_config, debug=debug)

    results_df = results.get_dataframe()
    print(results_df)
    if not debug:
        fp = f"tune_results_{uid}_{model_name}_{search_alg}.csv"
        print(f"[INFO] writing completed results to {fp}")
        results_df.to_csv(fp, index=False)

    # call fetch_postprocessed() just to match the full training loop
    _ = fetch_postprocessed_results_df(results)
    return


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache_dir", default="tmp",
                        help="Directory to cache raw data files to.")
    parser.add_argument("--cpu_per_worker", default=0, type=int,
                        help="Number of CPUs to provide per worker."
                             "If not set, Ray defaults to 1.")
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
    parser.add_argument("--num_workers", type=int, default=1,
                        help="Number of workers to use.")
    parser.add_argument("--no_tune", action="store_true", default=False,
                        help="If set, suppresses hyperparameter tuning of the "
                             "model (for faster testing).")
    parser.add_argument("--ray_local_dir", default=None, type=str,
                        help="""Set the local_dir argument to ray RunConfig. 
                        This is a local  directory where training results are 
                        saved to. If not specified, the script will first 
                        look for any of the dirs specified in ray_configs.py, 
                        and if none of those exist, it will use the Ray 
                        default.""")
    parser.add_argument("--ray_tmp_dir", default=None, type=str,
                        help="""Set the the root temporary path for ray. This 
                        is a local  directory where training results are 
                        saved to. If not specified, the script will first 
                        look for any of the dirs specified in ray_configs.py, 
                        and if none of those exist, it will use the Ray 
                        default of /tmp/ray. See 
                        https://docs.ray.io/en/latest/ray-core 
                        /configure.html#logging-and-debugging for more 
                        info.""")
    parser.add_argument("--scheduler", choices=(None, "asha", "median"),
                        default="asha",
                        help="Scheduler to use for hyperparameter optimization."
                             "See https://docs.ray.io/en/latest/tune/api_docs/schedulers.html .")

    parser.add_argument("--search_alg", default="hyperopt",
                        choices=["hyperopt", "random"],
                        help="Ray search alg to use for hyperparameter tuning.")
    parser.add_argument(
        "--uid",
        default=None,
        help="UID for experiment to run. Overridden when debug=True.")
    parser.add_argument("--use_cached", default=False, action="store_true",
                        help="whether to use cached data.")
    args = parser.parse_args()
    main(**vars(args))

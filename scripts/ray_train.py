import argparse
import logging
import os
from typing import Optional, List

import pandas as pd
import torch

from tablebench.core import CachedDataset
from tablebench.models.ray_utils import RayExperimentConfig, \
    run_ray_tune_experiment, \
    accuracy_metric_name_and_mode_for_model, \
    fetch_postprocessed_results_df
from tablebench.configs.experiment_configs import EXPERIMENT_CONFIGS
from tablebench.core import TabularDataset, TabularDatasetConfig
from tablebench.core.utils import timestamp_as_int
from tablebench.configs.ray_configs import get_default_ray_tmp_dir, \
    get_default_ray_local_dir
from tablebench.models.compat import PYTORCH_MODEL_NAMES, DOMAIN_GENERALIZATION_MODEL_NAMES

LOG_LEVEL = logging.DEBUG

logger = logging.getLogger()
logging.basicConfig(
    format='%(asctime)s %(levelname)-8s [%(filename)s:%(lineno)d] %(message)s',
    level=LOG_LEVEL,
    datefmt='%Y-%m-%d %H:%M:%S')


def main(experiment: str, uid: str, cache_dir: str,
         ray_tmp_dir: str,
         ray_local_dir: str,
         debug: bool,
         no_tune: bool, num_samples: int, search_alg: str,
         use_cached: bool,
         results_dir: str,
         models: Optional[List[str]] = None,
         max_concurrent_trials=2,
         num_workers=1,
         gpu_per_worker: float = 1.0,
         cpu_per_worker: int = 1,
         scheduler: str = None,
         gpu_models_only: bool = False,
         cpu_models_only: bool = False,
         no_dg: bool = False,
         exclude_models: Optional[List[str]] = None
         ):
    start_time = timestamp_as_int()
    assert not (gpu_models_only and cpu_models_only)
    if gpu_models_only:
        models = PYTORCH_MODEL_NAMES
        assert torch.cuda.is_available(), \
            "gpu_models_only is True but GPU is not available."
    elif cpu_models_only:
        models = ["xgb", "lightgbm"]
    else:
        assert models is not None

    if no_dg:
        logging.info(f"no_dg is {no_dg}; dropping domain generalization models")
        models = list(set(models) - set(DOMAIN_GENERALIZATION_MODEL_NAMES))

    if exclude_models:
        logging.info(f"dropping models {exclude_models}")
        models = list(set(models) - set(exclude_models))

    logging.info(f"training models {models}")

    if not ray_tmp_dir:
        ray_tmp_dir = get_default_ray_tmp_dir()
    if not ray_local_dir:
        ray_local_dir = get_default_ray_local_dir()

    if debug:
        logging.info("running in debug mode.")
        experiment = "_debug"

    if use_cached:
        logging.info(f"loading cached data from {cache_dir}")
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

    logging.debug(f"torch.cuda.is_available(): {torch.cuda.is_available()}")

    expt_results_dir = os.path.join(results_dir, experiment, str(start_time))
    logging.info(f"results will be written to {expt_results_dir}")
    if not os.path.exists(expt_results_dir): os.makedirs(expt_results_dir)

    iterates = []
    for model_name in models:
        logging.info(f"training model {model_name}")
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

        try:
            results = run_ray_tune_experiment(dset=dset, model_name=model_name,
                                              tune_config=tune_config,
                                              debug=debug)

            df = fetch_postprocessed_results_df(results)

            df["estimator"] = model_name
            df["domain_split_varname"] = dset.domain_split_varname
            df["domain_split_ood_values"] = str(dset.get_domains("ood_test"))
            df["domain_split_id_values"] = str(dset.get_domains("id_test"))
            if not debug:
                iter_fp = os.path.join(
                    expt_results_dir,
                    f"tune_results_{experiment}_{start_time}_{uid}_"
                    f"{model_name}.csv")
                logging.info(f"writing results for {model_name} to {iter_fp}")
                df.to_csv(iter_fp, index=False)
            iterates.append(df)

            print(df)
            logging.info(f"finished training model {model_name}")
        except Exception as e:
            logging.warning(f"exception training {model_name}: {e}, skipping")
            continue
    fp = os.path.join(expt_results_dir,
                      f"tune_results_{experiment}_{start_time}_full.csv")
    logging.info(f"writing results to {fp}")
    pd.concat(iterates).to_csv(fp, index=False)
    logging.info(f"completed domain shift experiment {experiment}!")
    return


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache_dir", default="tmp",
                        help="Directory to cache raw data files to.")
    parser.add_argument("--cpu_models_only", default=False,
                        action="store_true",
                        help="whether to only use models that use CPU."
                             "Mutually exclusive of --gpu_models_only.")
    parser.add_argument("--cpu_per_worker", default=0, type=int,
                        help="Number of CPUs to provide per worker."
                             "If not set, Ray defaults to 1.")
    parser.add_argument("--debug", action="store_true", default=False,
                        help="Whether to run in debug mode. If True, various "
                             "truncations/simplifications are performed to "
                             "speed up experiment.")
    parser.add_argument("--experiment", default="diabetes_readmission",
                        help="Experiment to run. Overridden when debug=True.")
    parser.add_argument("--exclude_models", nargs="+", action="store", default=[],
                        help="models to exclude, by name. Can be specified"
                             "multiple times, e.g. --exclude_models dro mlp xgb ...")
    parser.add_argument("--gpu_models_only", default=False,
                        action="store_true",
                        help="whether to only train models that use GPU."
                             "Mutually exclusive of cpu_models_only.")
    parser.add_argument("--gpu_per_worker", default=1.0, type=float,
                        help="GPUs per worker. Use fractional values < 1. "
                             "(e.g. --gpu_per_worker=0.5) in order"
                             "to allow multiple workers to share GPU.")
    parser.add_argument("--models", nargs="+", action="store", default=["mlp"],
                        help="Model names to train. Not used if "
                             "--cpu_models_only or --gpu_models_only is used."
                             "Can be specified multiple times, e.g."
                             "==model mlp xgb dro ...")
    parser.add_argument("--num_samples", type=int, default=100,
                        help="Number of hparam samples to take in tuning "
                             "sweep.")
    parser.add_argument("--num_workers", type=int, default=1,
                        help="Number of workers to use.")
    parser.add_argument("--no_dg", action="store_true", default=False,
                        help="If true, do NOT train domain generalization"
                             "models. Set this flag when there is only a"
                             "single training domain.")
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
    parser.add_argument("--results_dir", default="./ray_train_results",
                        help="where to write results. CSVs will be written to "
                             "experiment-specific subdirectories within this "
                             "directory.")
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

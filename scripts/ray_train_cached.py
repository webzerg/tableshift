import argparse

from tablebench.core import CachedDataset
from tablebench.models.ray_utils import TuneConfig, run_ray_tune_experiment


def main(experiment: str, uid: str, model_name: str, cache_dir: str,
         debug: bool,
         no_tune: bool, num_samples: int,
         tune_metric_name: str = "validation_accuracy",
         tune_metric_higher_is_better: bool = True,
         max_concurrent_trials=2,
         num_workers=1,
         early_stop=True):
    dset = CachedDataset(cache_dir=cache_dir, name=experiment, uid=uid)

    # dataset_config = TabularDatasetConfig(cache_dir=cache_dir)
    # tabular_dataset_kwargs = expt_config.tabular_dataset_kwargs
    # if "name" not in tabular_dataset_kwargs:
    #     tabular_dataset_kwargs["name"] = experiment
    #
    # dset = TabularDataset(config=dataset_config,
    #                       splitter=expt_config.splitter,
    #                       grouper=expt_config.grouper,
    #                       preprocessor_config=expt_config.preprocessor_config,
    #                       **tabular_dataset_kwargs)

    tune_config = TuneConfig(
        early_stop=early_stop,
        max_concurrent_trials=max_concurrent_trials,
        num_workers=num_workers,
        num_samples=num_samples,
        tune_metric_name=tune_metric_name,
        tune_metric_higher_is_better=tune_metric_higher_is_better,
    ) if not no_tune else None

    results = run_ray_tune_experiment(dset=dset, model_name=model_name,
                                      tune_config=tune_config, debug=debug)

    results_df = results.get_dataframe()
    print(results_df)
    results_df.to_csv(f"tune_results_{experiment}_{model_name}.csv",
                      index=False)
    print(results.get_best_result())
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
    parser.add_argument("--uid",
                        default="diabetes_readmissiondomain_split_varname_admission_type_iddomain_split_ood_value_1",
                        help="UID for experiment to run. Overridden when debug=True.")

    args = parser.parse_args()
    main(**vars(args))

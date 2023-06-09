import argparse
from tableshift.core import get_dataset

from tableshift.models.utils import get_estimator
from tableshift.models.training import train
from tableshift.configs.domain_shift import domain_shift_experiment_configs
from tableshift.models.config import get_default_config


def main(experiment, cache_dir, model, debug: bool):
    if debug:
        print("[INFO] running in debug mode.")
        experiment = "_debug"

    dset = get_dataset(name=experiment, cache_dir=cache_dir)
    config = get_default_config(model, dset)
    estimator = get_estimator(model, **config)
    train(estimator, dset, device="cpu", config=config)
    return


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache_dir", default="tmp",
                        help="Directory to cache raw data files to.")
    parser.add_argument("--debug", action="store_true", default=False,
                        help="Whether to run in debug mode. If True, various "
                             "truncations/simplifications are performed to "
                             "speed up experiment.")
    parser.add_argument("--experiment", choices=list(domain_shift_experiment_configs.keys()),
                        default="_debug",
                        help="Experiment to run. Overridden when debug=True.")
    parser.add_argument("--model", default="mlp",
                        help="model to use.")
    args = parser.parse_args()
    main(**vars(args))

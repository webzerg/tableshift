import argparse
from tablebench.core import TabularDataset, TabularDatasetConfig

from tablebench.datasets.experiment_configs import EXPERIMENT_CONFIGS
from tablebench.models import get_estimator, get_pytorch_model_config
from tablebench.models.training import _train_pytorch


def main(experiment: str, device: str, model: str, cache_dir: str):
    expt_config = EXPERIMENT_CONFIGS[experiment]

    dataset_config = TabularDatasetConfig(cache_dir=cache_dir)
    dset = TabularDataset(experiment,
                          config=dataset_config,
                          splitter=expt_config.splitter,
                          grouper=expt_config.grouper,
                          preprocessor_config=expt_config.preprocessor_config,
                          **expt_config.tabular_dataset_kwargs)

    config = get_pytorch_model_config(model, dset)
    model = get_estimator(model, **config)

    _train_pytorch(model, dset, device)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache_dir", default="tmp",
                        help="Directory to cache raw data files to.")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--experiment", default="adult")
    parser.add_argument("--model", default="mlp", choices=(
        "ft_transformer", "mlp", "resnet", "group_dro"))
    args = parser.parse_args()
    main(**vars(args))

import argparse
from tablebench.core import TabularDataset, TabularDatasetConfig

from tablebench.datasets.experiment_configs import EXPERIMENT_CONFIGS
from tablebench.models import get_estimator
from tablebench.models.training import train_pytorch


def main(experiment: str, device: str, model: str, cache_dir: str):
    expt_config = EXPERIMENT_CONFIGS[experiment]

    dataset_config = TabularDatasetConfig(cache_dir=cache_dir)
    dset = TabularDataset(experiment,
                          config=dataset_config,
                          splitter=expt_config.splitter,
                          grouper=expt_config.grouper,
                          preprocessor_config=expt_config.preprocessor_config,
                          **expt_config.tabular_dataset_kwargs)

    # TODO(jpgard): set all architectural defaults here
    #  based on [gorishniy2021revisiting] paper.
    # A default set of arguments for each model. Note: these could be
    # bad choices for defaults; they are strictly for testing!
    model_args = {
        "ft_transformer": dict(n_num_features=dset.X_shape[1],
                               cat_cardinalities=None),
        "mlp": dict(d_in=dset.X_shape[1], d_layers=[256, 256]),
        "resnet": dict(d_in=dset.X_shape[1]),
        "group_dro": dict(d_in=dset.X_shape[1], d_layers=[256, 256],
                          n_groups=dset.n_groups, group_weights_step_size=0.05),
    }
    model = get_estimator(model, **model_args[model])

    train_pytorch(model, dset, device)


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

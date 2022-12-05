import argparse
from functools import partial
from typing import Dict, Tuple

import numpy as np
import pandas as pd
import ray
from ray import tune
from ray.tune import Tuner
from ray.tune.schedulers import ASHAScheduler
from ray.tune.search.hyperopt import HyperOptSearch
from ray.air.config import RunConfig
from ray.train.torch import TorchTrainer
from ray.air.config import ScalingConfig
from ray import train
from ray.air import session
from ray.train.torch import TorchCheckpoint
import rtdl
import scipy
import sklearn
import torch
import torch.nn.functional as F

from tablebench.core import TabularDataset, TabularDatasetConfig
from tablebench.datasets.experiment_configs import EXPERIMENT_CONFIGS
from tablebench.models import get_estimator, get_model_config
from tablebench.models.compat import SklearnStylePytorchModel
from tablebench.models.training import get_optimizer, get_criterion


def make_ray_dataset(dset: TabularDataset, split):
    X, y, G, _ = dset.get_pandas(split)
    df = pd.concat([X, y, G], axis=1)
    df = df.loc[:, ~df.columns.duplicated()].copy()

    dataset: ray.data.Dataset = ray.data.from_pandas([df])
    return dataset


@torch.no_grad()
def get_predictions_and_labels(model, loader, as_logits=False) -> Tuple[
    np.ndarray, np.ndarray]:
    """Get the predictions (as logits, or probabilities) and labels."""
    prediction = []
    label = []

    for batch in loader:
        batch_x, batch_y = batch["x"], batch["y"]
        batch_x = batch_x.float()
        batch_y = batch_y.float()
        # TODO(jpgard): handle categorical features here.
        prediction.append(model(batch_x))
        label.append(batch_y)
    prediction = torch.cat(prediction).squeeze().cpu().numpy()
    target = torch.cat(label).squeeze().cpu().numpy()
    if not as_logits:
        prediction = scipy.special.expit(prediction)
    return prediction, target


def ray_train_epoch(model, optimizer, criterion, train_loader,
                    epoch: int) -> float:
    """Run one epoch of training, and return the training loss."""
    print(f"starting epoch {epoch}")

    model.train()
    running_loss = 0.0
    n_train = 0
    for i, batch in enumerate(train_loader):
        # get the inputs and labels
        inputs, labels, groups = batch["x"], batch["y"], batch["g"]
        inputs = inputs.float()
        labels = labels.float()

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = model(inputs).squeeze()
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # print statistics
        running_loss += loss.item()
        n_train += len(inputs)
        if i % 2000 == 1999:  # print every 2000 mini-batches
            print(
                f"[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 2000:.3f}")
            running_loss = 0.0
    return running_loss / n_train


def ray_evaluate(model, eval_batches) -> dict:
    """Run evaluation of a model.

    eval_batches should be a dict mapping split names to DataLoaders.
    """
    model.eval()
    prediction, target = get_predictions_and_labels(
        model, eval_batches["validation"])
    prediction = np.round(prediction)
    val_acc = sklearn.metrics.accuracy_score(target, prediction)
    return dict(validation_accuracy=val_acc)


def main(experiment: str, device: str, model_name: str, cache_dir: str,
         debug: bool,
         no_tune: bool, num_samples: int,
         tune_metric_name: str = "validation_accuracy",
         tune_metric_higher_is_better: bool = True,
         max_concurrent_trials=2):
    if debug:
        print("[INFO] running in debug mode.")
        experiment = "_debug"
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

    X, y, G, _ = dset.get_pandas("train")
    y_name = y.name
    G_names = G.columns.tolist()
    X_names = X.columns.tolist()

    def _row_to_dict(row) -> Dict:
        """Convert ray PandasRow to a dict of numpy arrays."""
        x = row[X_names].values.astype(float)
        y = row[y_name].values.astype(float)
        g = row[G_names].values.astype(float)
        return {"x": x, "y": y, "g": g}

    train_dataset = make_ray_dataset(dset, "train")
    train_dataset = train_dataset.map_batches(_row_to_dict,
                                              batch_format="pandas")

    val_dataset = make_ray_dataset(dset, "validation")
    val_dataset = val_dataset.map_batches(_row_to_dict, batch_format="pandas")

    def train_loop_per_worker(config: Dict):
        """Function to be run by each Trainer.

        Must be defined inside main() because this function can only have a
        single argument, named config, but it also requires the use of the
        model_name command-line flag.
        """
        model = get_estimator(model_name, d_in=config["d_in"],
                              d_layers=[config["d_hidden"]] * config[
                                  "num_layers"])
        assert isinstance(model, SklearnStylePytorchModel)
        model = train.torch.prepare_model(model)

        criterion = get_criterion(model)
        optimizer = get_optimizer(model, config)

        train_dataset_shard = session.get_dataset_shard("train")
        val_dataset_shard = session.get_dataset_shard("validation")

        # Returns the current torch device; useful for sending to a device.
        # train.torch.get_device()
        train_dataset_batches = train_dataset_shard.iter_torch_batches(
            batch_size=config["batch_size"])
        eval_batches = {
            "validation": val_dataset_shard.iter_torch_batches(
                batch_size=config["batch_size"]),
        }
        for epoch in range(config["n_epochs"]):
            train_loss = ray_train_epoch(model, optimizer, criterion,
                                         train_dataset_batches,
                                         epoch)
            metrics = ray_evaluate(model, eval_batches)

            # Log the metrics for this epoch
            metrics.update(dict(train_loss=train_loss))
            checkpoint = TorchCheckpoint.from_state_dict(
                model.module.state_dict())
            session.report(metrics, checkpoint=checkpoint)

    # Get the default configs
    default_train_config = get_model_config(model_name, dset)
    # Trainer object that will be passed to each worker.
    trainer = TorchTrainer(
        train_loop_per_worker=train_loop_per_worker,
        train_loop_config=default_train_config,
        datasets={"train": train_dataset, "validation": val_dataset},
        scaling_config=ScalingConfig(num_workers=2,
                                     use_gpu=torch.cuda.is_available()),
    )

    if no_tune:
        # To run just a single training iteration (without tuning)
        result = trainer.fit()
        latest_checkpoint = result.checkpoint
        return

    # Hyperparameter search space; note that the scaling_config can also be tuned
    # but is fixed here.
    param_space = {
        # The params will be merged with the ones defined in the TorchTrainer
        "train_loop_config": {
            # This is a parameter that hasn't been set in the TorchTrainer
            "num_layers": tune.randint(1, 4),
            "lr": tune.loguniform(1e-4, 1e-1),
            "weight_decay": tune.loguniform(1e-4, 1e0),
            "d_hidden": tune.choice([64, 128, 256, 512]),
        },
        # Tune the number of distributed workers
        "scaling_config": ScalingConfig(num_workers=2),

        # Note: when num_workers=1, trials seemed to fail with AttributeError
        # (MLPModel does not have attribute 'module'); not sure why.
        # "scaling_config": ScalingConfig(num_workers=tune.grid_search([1, 2])),
    }

    # Create Tuner
    mode = "max" if tune_metric_higher_is_better else "min"
    tuner = Tuner(
        trainable=trainer,
        run_config=RunConfig(name="test_tuner_notebook",
                             local_dir="ray-results"),
        param_space=param_space,
        tune_config=tune.TuneConfig(
            search_alg=HyperOptSearch(metric=tune_metric_name, mode=mode),
            scheduler=ASHAScheduler(
                time_attr='training_iteration',
                metric=tune_metric_name,
                mode=mode,
                stop_last_trials=True),
            num_samples=num_samples,
            max_concurrent_trials=max_concurrent_trials),
    )

    results = tuner.fit()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache_dir", default="tmp",
                        help="Directory to cache raw data files to.")
    parser.add_argument("--debug", action="store_true", default=False,
                        help="Whether to run in debug mode. If True, various "
                             "truncations/simplifications are performed to "
                             "speed up experiment.")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--experiment", default="adult",
                        help="Experiment to run. Overridden when debug=True.")
    parser.add_argument("--model_name", default="mlp")
    parser.add_argument("--num_samples", type=int, default=1,
                        help="Number of hparam samples to take in tuning "
                             "sweep.")
    parser.add_argument("--no_tune", action="store_true", default=False,
                        help="If set, suppresses hyperparameter tuning of the "
                             "model (for faster testing).")
    args = parser.parse_args()
    main(**vars(args))

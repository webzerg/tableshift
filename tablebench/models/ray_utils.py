from dataclasses import dataclass
from functools import partial
from typing import Dict, Any, List

import numpy as np
import pandas as pd
import ray
import sklearn
import torch
from ray import train
from ray.train.torch import TorchCheckpoint

from tablebench.core import TabularDataset
from tablebench.models.torchutils import get_predictions_and_labels


@dataclass
class TuneConfig:
    """Container for various Ray tuning parameters.

    Note that this is different from the Ray TuneConfig class, as it actually
    contains parameters that are passed to different parts of the ray API
    such as `ScalingConfig`, which consumes the num_workers."""
    max_concurrent_trials: int
    num_workers: int = 1
    num_samples: int = 1
    tune_metric_name: str = "metric"
    tune_metric_higher_is_better: bool = True
    early_stop: bool = True

    @property
    def mode(self):
        return "max" if self.tune_metric_higher_is_better else "min"


def make_ray_dataset(dset: TabularDataset, split, keep_domain_labels=False):
    X, y, G, d = dset.get_pandas(split)
    if (d is None) or (not keep_domain_labels):
        df = pd.concat([X, y, G], axis=1)
    else:
        df = pd.concat([X, y, G, d], axis=1)
    df = df.loc[:, ~df.columns.duplicated()].copy()

    dataset: ray.data.Dataset = ray.data.from_pandas([df])
    return dataset


def get_ray_checkpoint(model):
    if isinstance(model, torch.nn.parallel.DistributedDataParallel):
        return TorchCheckpoint.from_state_dict(model.module.state_dict())
    else:
        return TorchCheckpoint.from_state_dict(model.state_dict())


def ray_evaluate(model, splits: Dict[str, Any]) -> dict:
    """Run evaluation of a model.

    splits should be a dict mapping split names to DataLoaders.
    """
    device = train.torch.get_device()
    model.eval()
    metrics = {}
    for split in splits:
        prediction, target = get_predictions_and_labels(model, splits[split],
                                                        device=device)
        prediction = np.round(prediction)
        acc = sklearn.metrics.accuracy_score(target, prediction)
        metrics[f"{split}_accuracy"] = acc
    return metrics


def _row_to_dict(row, X_names: List[str], y_name: str, G_names: List[str],
                 d_name: str) -> Dict:
    """Convert ray PandasRow to a dict of numpy arrays."""
    x = row[X_names].values.astype(float)
    y = row[y_name].values.astype(float)
    g = row[G_names].values.astype(float)
    outputs = {"x": x, "y": y, "g": g}
    if d_name in row:
        outputs["d"] = row[d_name].values.astype(float)
    return outputs


def prepare_torch_datasets(split, dset: TabularDataset):
    keep_domain_labels = dset.domain_label_colname is not None
    ds = make_ray_dataset(dset, split, keep_domain_labels)

    y_name = dset.target
    d_name = dset.domain_label_colname
    G_names = dset.group_feature_names
    X_names = dset.feature_names

    _map_fn = partial(_row_to_dict, X_names=X_names, y_name=y_name,
                      G_names=G_names, d_name=d_name)

    return ds.map_batches(_map_fn, batch_format="pandas")

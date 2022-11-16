import argparse
import rtdl
import torch
import torch.nn.functional as F
from tablebench.core import TabularDataset, TabularDatasetConfig

from tablebench.datasets.experiment_configs import EXPERIMENT_CONFIGS
from tablebench.models import get_estimator

experiment = "adult"
expt_config = EXPERIMENT_CONFIGS[experiment]

dataset_config = TabularDatasetConfig()
dset = TabularDataset(experiment,
                      config=dataset_config,
                      splitter=expt_config.splitter,
                      grouper=expt_config.grouper,
                      preprocessor_config=expt_config.preprocessor_config,
                      **expt_config.tabular_dataset_kwargs)
train_loader = dset.get_dataloader("train", 512)
loaders = {s: dset.get_dataloader(s, 2048) for s in ("validation", "test")}

model = get_estimator("mlp", d_in=dset.X_shape[1], d_layers=[256, 256])
# model = get_estimator("resnet", d_in=dset.X_shape[1])

lr = 0.001
weight_decay = 0.0
loss_fn = F.binary_cross_entropy_with_logits

optimizer = (
    model.make_default_optimizer()
    if isinstance(model, rtdl.FTTransformer)
    else torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
)
model.fit(train_loader, optimizer, loss_fn, n_epochs=2, other_loaders=loaders)

import argparse
import rtdl
import torch
import torch.nn.functional as F
from tablebench.core import TabularDataset, TabularDatasetConfig

from tablebench.datasets.experiment_configs import EXPERIMENT_CONFIGS
from tablebench.models import get_estimator
from tablebench.models.dro import group_dro_loss, GroupDROModel

experiment = "adult"
expt_config = EXPERIMENT_CONFIGS[experiment]
device = "cpu"

dataset_config = TabularDatasetConfig()
dset = TabularDataset(experiment,
                      config=dataset_config,
                      splitter=expt_config.splitter,
                      grouper=expt_config.grouper,
                      preprocessor_config=expt_config.preprocessor_config,
                      **expt_config.tabular_dataset_kwargs)
train_loader = dset.get_dataloader("train", 512, device=device)
loaders = {s: dset.get_dataloader(s, 2048) for s in ("validation", "test")}
#
# model = get_estimator("ft_transformer",
#                       # TODO(jpgard): set these to defaults from the
#                       #  [gorishniy2021revisiting] paper.
#                       d_token=8,
#                       n_blocks=3,
#                       attention_dropout=0.,
#                       ffn_d_hidden=1,
#                       ffn_dropout=0.,
#                       residual_dropout=0.,
#                       cat_cardinalities=None,
#                       n_num_features=dset.X_shape[1],
#                       device=device)
model = get_estimator("mlp", d_in=dset.X_shape[1], d_layers=[256, 256],
                      device=device)
# model = get_estimator("resnet", d_in=dset.X_shape[1], device=device)
# model = get_estimator("group_dro", d_in=dset.X_shape[1], d_layers=[256, 256],
#                       n_groups=dset.n_groups, group_weights_step_size=0.05,
#                       device=device)

lr = 0.001
weight_decay = 0.0
loss_fn = (group_dro_loss
           if isinstance(model, GroupDROModel)
           else F.binary_cross_entropy_with_logits)

optimizer = (
    model.make_default_optimizer()
    if isinstance(model, rtdl.FTTransformer)
    else torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
)
model.fit(train_loader, optimizer, loss_fn, n_epochs=2, other_loaders=loaders)

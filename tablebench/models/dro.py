from typing import List, Optional, Mapping, Callable, Type

import torch

from tablebench.models.compat import SklearnStylePytorchModel
from tablebench.models.rtdl import MLPModel
from tablebench.models.torchutils import unpack_batch, apply_model
from tablebench.models.losses import GroupDROLoss


class GroupDROModel(MLPModel, SklearnStylePytorchModel):
    def __init__(self, group_weights_step_size: float, n_groups: int, **kwargs):
        MLPModel.__init__(self, **kwargs)

        assert n_groups > 0, "require nonzero n_groups."
        self.group_weights_step_size = torch.Tensor([group_weights_step_size])
        # initialize adversarial weights
        self.group_weights = torch.nn.Parameter(torch.full([n_groups], 1. / n_groups))

    def to(self, device):
        super().to(device)
        for attr in ("group_weights_step_size", "group_weights"):
            setattr(self, attr, getattr(self, attr).to(device))
        return self

    def train_epoch(self, train_loader: torch.utils.data.DataLoader,
                    optimizer: torch.optim.Optimizer,
                    loss_fn: GroupDROLoss,
                    device: str,
                    other_loaders: Optional[
                        Mapping[str, torch.utils.data.DataLoader]] = None
                    ):
        for iteration, batch in enumerate(train_loader):
            x_batch, y_batch, _, d_batch = unpack_batch(batch)
            self.train()
            optimizer.zero_grad()
            outputs = apply_model(self, x_batch)
            loss = loss_fn(outputs.squeeze(1), y_batch, d_batch,
                           self.group_weights,
                           self.group_weights_step_size)

            loss.backward()
            optimizer.step()

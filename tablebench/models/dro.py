import itertools
from typing import List, Optional, Mapping, Callable, Type

import rtdl
import torch
from torch.nn.functional import binary_cross_entropy_with_logits

from tablebench.models.compat import SklearnStylePytorchModel
from tablebench.models.rtdl import MLPModel
from tablebench.models.utils import evaluate, apply_model


def group_dro_loss(outputs: torch.Tensor, targets: torch.Tensor,
                   sens: torch.Tensor):
    assert torch.all(torch.logical_or(sens == 0., sens == 1.)), \
        "only binary groups supported."
    subgroup_losses = []
    # TODO(jpgard): check that sens is a binary matrix (only 0/1).
    n_attrs = sens.shape[1]
    elementwise_loss = binary_cross_entropy_with_logits(input=outputs,
                                                        target=targets,
                                                        reduction="none")
    # Compute the loss on each subgroup
    for subgroup_idxs in itertools.product(*[(0, 1)] * n_attrs):
        subgroup_idxs = torch.Tensor(subgroup_idxs).to(sens.device)
        mask = torch.all(sens == subgroup_idxs, dim=1)
        subgroup_loss = elementwise_loss[mask].sum() / mask.sum()
        subgroup_losses.append(subgroup_loss)
    return torch.stack(subgroup_losses)


class GroupDROModel(MLPModel, SklearnStylePytorchModel):
    def __init__(self, group_weights_step_size: float, n_groups: int,
                 **kwargs):
        MLPModel.__init__(self, **kwargs)
        # TODO(jpgard): send these Tensors to device.
        self.group_weights_step_size = torch.Tensor([group_weights_step_size])
        # initialize adversarial weights
        self.group_weights = torch.full([n_groups], 1. / n_groups)

    @classmethod
    def make_baseline(cls: Type['GroupDROModel'],
                      d_in: int,
                      d_layers: List[int],
                      dropout: float,
                      d_out: int,
                      group_weights_step_size: float,
                      n_groups: int) -> 'GroupDROModel':
        """Create a baseline Group DRO model."""
        assert isinstance(dropout,
                          float), 'In this constructor, dropout must be float'
        if len(d_layers) > 2:
            assert len(set(d_layers[1:-1])) == 1, (
                'In this constructor, if d_layers contains more than two '
                'elements, then all elements except for the first and the '
                'last ones must be equal. '
            )
        return cls(
            d_in=d_in,
            d_layers=d_layers,  # type: ignore
            dropouts=dropout,
            activation='ReLU',
            d_out=d_out,
            group_weights_step_size=group_weights_step_size,
            n_groups=n_groups
        )

    def train_epoch(self, train_loader: torch.utils.data.DataLoader,
                    optimizer: torch.optim.Optimizer,
                    loss_fn: Callable,
                    other_loaders: Optional[
                        Mapping[str, torch.utils.data.DataLoader]] = None
                    ):
        for iteration, (x_batch, y_batch, g_batch) in enumerate(train_loader):
            self.train()
            optimizer.zero_grad()
            outputs = apply_model(self, x_batch)
            group_losses = loss_fn(outputs.squeeze(1), y_batch, g_batch)
            # update group weights
            self.group_weights = self.group_weights * torch.exp(
                self.group_weights_step_size * group_losses.data)
            self.group_weights = (
                    self.group_weights / (self.group_weights.sum()))
            # update model
            loss = group_losses @ self.group_weights
            loss.backward()
            optimizer.step()

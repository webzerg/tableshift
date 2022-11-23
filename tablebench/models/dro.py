import itertools
from typing import List, Optional, Mapping, Callable, Type

import rtdl
import torch
from torch.nn.functional import binary_cross_entropy_with_logits

from tablebench.models.compat import SklearnStylePytorchModel
from tablebench.models.rtdl import MLPModel
from tablebench.models.utils import apply_model, unpack_batch


# TODO(jpgard): make this a loss object, with n_groups as an attribute.
def group_dro_loss(outputs: torch.Tensor, targets: torch.Tensor,
                   group_ids: torch.Tensor, n_groups: int) -> torch.Tensor:
    group_ids = group_ids.int()
    assert group_ids.max() < n_groups
    subgroup_losses = torch.zeros(n_groups, dtype=torch.float)

    elementwise_loss = binary_cross_entropy_with_logits(input=outputs,
                                                        target=targets,
                                                        reduction="none")
    # Compute the average loss on each subgroup present in the data.
    for group_id in torch.unique(group_ids):
        mask = (group_ids == group_id)
        subgroup_loss = elementwise_loss[mask].mean()
        subgroup_losses[group_id] = subgroup_loss

    return subgroup_losses


class GroupDROModel(MLPModel, SklearnStylePytorchModel):
    def __init__(self, group_weights_step_size: float, n_groups: int, **kwargs):
        MLPModel.__init__(self, **kwargs)

        self.group_weights_step_size = torch.Tensor([group_weights_step_size])
        # initialize adversarial weights
        self.group_weights = torch.full([n_groups], 1. / n_groups)

    def to(self, device):
        super().to(device)
        for attr in ("group_weights_step_size", "group_weights"):
            setattr(self, attr, getattr(self, attr).to(device))

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
        for iteration, batch in enumerate(train_loader):
            x_batch, y_batch, _, d_batch = unpack_batch(batch)
            self.train()
            optimizer.zero_grad()
            outputs = apply_model(self, x_batch)
            group_losses = loss_fn(outputs.squeeze(1), y_batch, d_batch,
                                   n_groups=len(self.group_weights))
            # update group weights
            self.group_weights = self.group_weights * torch.exp(
                self.group_weights_step_size * group_losses.data)
            self.group_weights = (
                    self.group_weights / (self.group_weights.sum()))
            # update model
            loss = group_losses @ self.group_weights
            loss.backward()
            optimizer.step()

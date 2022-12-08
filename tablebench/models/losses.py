from dataclasses import dataclass

from torch.nn.functional import binary_cross_entropy_with_logits
import torch


@dataclass
class DomainLoss:
    """A class to represent losses that require domain labels."""

    @classmethod
    def __call__(self, *args, **kwargs):
        raise


@dataclass
class GroupDROLoss(DomainLoss):
    n_groups: int

    def __call__(self, outputs: torch.Tensor,
                 targets: torch.Tensor, group_ids: torch.Tensor,
                 group_weights: torch.Tensor,
                 group_weights_step_size: torch.Tensor,
                 device):
        """Compute the Group DRO objective."""
        group_ids = group_ids.int()
        assert group_ids.max() < self.n_groups

        group_losses = torch.zeros(self.n_groups, dtype=torch.float,
                                   device=device)

        elementwise_loss = binary_cross_entropy_with_logits(input=outputs,
                                                            target=targets,
                                                            reduction="none")
        # Compute the average loss on each subgroup present in the data.
        for group_id in torch.unique(group_ids):
            mask = (group_ids == group_id)
            subgroup_loss = elementwise_loss[mask].mean()
            group_losses[group_id] = subgroup_loss

        # update group weights
        group_weights = group_weights * torch.exp(
            group_weights_step_size * group_losses.data)
        group_weights = (group_weights / (group_weights.sum()))

        return group_losses @ group_weights

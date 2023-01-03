from typing import Callable
from dataclasses import dataclass

from torch.nn.functional import binary_cross_entropy_with_logits
import torch
from torch import Tensor
from tablebench.models.fastdro.robust_losses import RobustLoss


def irm_penalty(logits: Tensor, y: Tensor) -> Tensor:
    """IRM penalty term. Adapted from DomainBed.

    See https://github.com/facebookresearch/DomainBed/blob/main/domainbed
    /algorithms.py.
    """
    device = "cuda" if logits[0][0].is_cuda else "cpu"
    scale = torch.tensor(1.).to(device).requires_grad_()
    loss_1 = F.cross_entropy(logits[::2] * scale, y[::2])
    loss_2 = F.cross_entropy(logits[1::2] * scale, y[1::2])
    grad_1 = autograd.grad(loss_1, [scale], create_graph=True)[0]
    grad_2 = autograd.grad(loss_2, [scale], create_graph=True)[0]
    result = torch.sum(grad_1 * grad_2)
    return result


class DROLoss(RobustLoss):
    """Robust loss that computes the DRO loss."""

    def __init__(self, base_loss_fn: Callable[
        [Tensor, Tensor], Tensor] = binary_cross_entropy_with_logits, **kwargs):
        assert kwargs["geometry"] in ("cvar", "chi-square")
        super().__init__(**kwargs)
        self.base_loss_fn = base_loss_fn

    def forward(self, input, target):
        elementwise_loss = self.base_loss_fn(input=input, target=target,
                                             reduction='none')
        assert len(elementwise_loss) == len(
            input), "(non-)reduction sanity check"
        return RobustLoss.forward(self, elementwise_loss)


@dataclass
class DomainLoss:
    """A class to represent losses that require domain labels."""

    @classmethod
    def __call__(self, *args, **kwargs):
        raise


@dataclass
class DomainGeneralizationLoss:
    """A class to represent losses for domain generalization."""


@dataclass
class CORALLoss(DomainGeneralizationLoss):

    def __call__(self, activations_id, activations_ood):
        assert activations_id.shape[1] == activations_ood.shape[1], \
            f"got unexpected activations shapes: {activations_id.shape}, " \
            f"{activations_ood.shape}"
        d = activations_id.shape[1]
        C_s = torch.cov(activations_id)
        C_t = torch.cov(activations_ood)
        const = 1 / (4 * d ** 2)
        dist = torch.norm(C_s - C_t, p="fro") ** 2
        return const * dist


@dataclass
class GroupDROLoss(DomainLoss):
    n_groups: int

    def __call__(self, outputs: Tensor,
                 targets: Tensor, group_ids: Tensor,
                 group_weights: Tensor,
                 group_weights_step_size: Tensor,
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

import copy
from typing import Callable, Mapping, Optional

import torch
import torch.nn.functional as F

from tablebench.models.losses import irm_penalty
from tablebench.models.compat import SklearnStylePytorchModel
from tablebench.models.rtdl import MLPModel
from tablebench.models.torchutils import unpack_batch, apply_model
from tablebench.models.utils import OPTIMIZER_ARGS


class IRMModel(MLPModel, SklearnStylePytorchModel):
    """Class to represent Invariant Risk Minimization models.

    Based on implementation from
    https://github.com/facebookresearch/DomainBed/blob/main/domainbed
    /algorithms.py .
    """

    def __init__(self, irm_lambda: float, irm_penalty_anneal_iters: int,
                 **hparams):
        self.config = copy.deepcopy(hparams)

        super().__init__(**hparams)

        self.irm_lambda = irm_lambda
        self.irm_penalty_anneal_iters = irm_penalty_anneal_iters
        self.register_buffer('update_count', torch.tensor([0]))

    def train_epoch(self, train_loader: torch.utils.data.DataLoader,
                    loss_fn: Callable,
                    device: str,
                    other_loaders: Optional[
                        Mapping[str, torch.utils.data.DataLoader]] = None,
                    ) -> float:
        """IRM training epoch.

        Implementation via https://github.com/facebookresearch/DomainBed/blob
        /main/domainbed/algorithms.py#L330.

        """
        penalty_weight = (
            self.irm_lambda
            if self.update_count >= self.irm_penalty_anneal_iters
            else 1.0)

        nll = torch.Tensor([0.])
        penalty = torch.Tensor([0.])
        num_batches = 0.

        for batch in train_loader:
            x_batch, y_batch, _, _ = unpack_batch(batch)
            self.train()
            self.optimizer.zero_grad()
            logits = apply_model(self, x_batch).squeeze(1)
            nll += F.cross_entropy(logits, y_batch)
            penalty += irm_penalty(logits, y_batch)
            num_batches += 1

        nll /= num_batches
        penalty /= num_batches
        loss = nll + (penalty_weight * penalty)

        if self.update_count == self.irm_penalty_anneal_iters:
            # Reset Adam, because it doesn't like the sharp jump in gradient
            # magnitudes that happens at this step.
            self._init_optimizer()

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.update_count += 1
        return loss.item()

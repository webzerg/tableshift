import copy
from typing import Callable, Mapping, Optional, Dict, Any, Union, Tuple

import torch
from torch import Tensor
from torch.utils.data import DataLoader
import torch.nn.functional as F
import torch.autograd as autograd

from tablebench.models.compat import SklearnStylePytorchModel
from tablebench.models.rtdl import MLPModel
from tablebench.models.torchutils import unpack_batch, apply_model


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

    @staticmethod
    def _irm_penalty(logits, y):
        device = "cuda" if logits.is_cuda else "cpu"
        scale = torch.tensor(1.).to(device).requires_grad_()
        loss_1 = F.cross_entropy(logits[::2] * scale, y[::2])
        loss_2 = F.cross_entropy(logits[1::2] * scale, y[1::2])
        grad_1 = autograd.grad(loss_1, [scale], create_graph=True)[0]
        grad_2 = autograd.grad(loss_2, [scale], create_graph=True)[0]
        result = torch.sum(grad_1 * grad_2)
        return result

    def update(self, minibatches, unlabeled=None):

        penalty_weight = (
            self.hparams['irm_lambda']
            if self.update_count >= self.irm_penalty_anneal_iters
            else 1.0)
        nll = 0.
        penalty = 0.
        all_x = torch.cat([x for x, y in minibatches])
        all_logits = apply_model(self, all_x).squeeze()
        all_logits_idx = 0

        for i, (x, y) in enumerate(minibatches):
            print(i)
            logits = all_logits[all_logits_idx:all_logits_idx + x.shape[0]]
            all_logits_idx += x.shape[0]
            nll += F.cross_entropy(logits, y)
            penalty += self._irm_penalty(logits, y)

        nll /= len(minibatches)
        penalty /= len(minibatches)
        loss = nll + (penalty_weight * penalty)

        if self.update_count == self.irm_penalty_anneal_iters:
            # Reset Adam, because it doesn't like the sharp jump in gradient
            # magnitudes that happens at this step.
            self._init_optimizer()

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.update_count += 1
        return {'loss': loss.item(), 'nll': nll.item(),
                'penalty': penalty.item()}

    def train_epoch(self,
                    train_loaders: Union[DataLoader, Dict[Any, DataLoader]],
                    loss_fn: Callable,
                    device: str,
                    uda_loader: Optional[DataLoader] = None,
                    other_loaders: Optional[Mapping[str, DataLoader]] = None,
                    steps: Optional[int] = None
                    ) -> float:
        """Conduct one epoch of training and return the loss."""

        loaders = [x for x in train_loaders.values()]
        train_minibatches_iterator = zip(*loaders)

        def _prepare_batch(batch) -> Tuple[Tensor, Tensor]:
            x_batch, y_batch = batch[:2]
            return x_batch.to(device), y_batch.to(device)

        print("[WARNING] remove 100-step limit after testing!!!")
        print("#" * 50)
        for step in range(100):

            minibatches_device = [_prepare_batch(batch) for batch in
                                  next(train_minibatches_iterator)]
            # Note: if this was a domain_adaption task, do the same as above
            # for uda_loader.
            tmp = self.update(minibatches_device)
            import ipdb;
            ipdb.set_trace()

import copy
from math import floor
from typing import Callable, Mapping, Optional

import numpy as np
import torch

from tablebench.models.compat import SklearnStylePytorchModel
from tablebench.models.rtdl import MLPModel
from tablebench.models.torchutils import unpack_batch, apply_model


def random_minibatch_idxs(batch_size, microbatch_size):
    """Based on DomainBed random_pairs_of_minibatches() but yields indices into a batch."""
    if batch_size % microbatch_size != 0:
        num_microbatches = int(floor(batch_size / microbatch_size))
        print(f"[DEBUG] batch_size {batch_size} not divisible by "
              f"microbatch_size {microbatch_size}; attempting to use partial "
              f"batch. (This is an expected message in the final batch of an "
              f"epoch when batch size does not evenly divide dataset size.)")
    else:
        num_microbatches = int(batch_size / microbatch_size)
    rng = np.random.default_rng()
    i = np.arange(batch_size)
    j = rng.permutation(batch_size)

    idxs = [(i[x:x + microbatch_size], j[x:x + microbatch_size])
            for x in range(num_microbatches)]
    return idxs


class MixUpModel(MLPModel, SklearnStylePytorchModel):
    """
    Class to train via Mixup of batches from different domains.

    Implementation via:
        https://github.com/facebookresearch/DomainBed/blob/main/domainbed/algorithms.py#L413
    Citations:
        https://arxiv.org/pdf/2001.00677.pdf
        https://arxiv.org/pdf/1912.01805.pdf
    """

    def __init__(self, mixup_alpha: float, **hparams):
        self.config = copy.deepcopy(hparams)

        super().__init__(**hparams)
        self.mixup_alpha = mixup_alpha

    def train_epoch(self, train_loaders: torch.utils.data.DataLoader,
                    loss_fn: Callable,
                    device: str,
                    eval_loaders: Optional[
                        Mapping[str, torch.utils.data.DataLoader]] = None,
                    ) -> float:
        total_loss = None
        n_train = 0
        microbatch_size = 16  # must evenly divide batch size

        for batch in train_loaders:
            loss = 0

            x_batch, y_batch, _, _ = unpack_batch(batch)
            batch_size = len(x_batch)

            for idxs_i, idxs_j in random_minibatch_idxs(batch_size,
                                                        microbatch_size):

                lam = np.random.beta(self.mixup_alpha, self.mixup_alpha)
                x = lam * x_batch[idxs_i] + (1 - lam) * x_batch[idxs_j]
                predictions = apply_model(self, x).squeeze()
                loss += lam * loss_fn(predictions, y_batch[idxs_i])
                loss += (1 - lam) * loss_fn(predictions, y_batch[idxs_j])

            loss /= batch_size
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            batch_loss = loss.detach().cpu().numpy().item()
            n_train += batch_size

            if total_loss is None:
                total_loss = batch_loss
            else:
                total_loss += batch_loss

        return total_loss / n_train

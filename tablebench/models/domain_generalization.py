from abc import abstractmethod
from typing import Union, Dict, Any, Callable, Optional, Mapping, Tuple

from torch import Tensor
from torch.utils.data import DataLoader

from tablebench.models.rtdl import MLPModel
from tablebench.models.torchutils import unpack_batch


class DomainGeneralizationModel(MLPModel):
    """Class to represent models trained for domain generalization."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.domain_generalization = True

    @abstractmethod
    def update(self, minibatches, unlabeled=None):
        raise

    def train_epoch(self,
                    train_loaders: Union[DataLoader, Dict[Any, DataLoader]],
                    loss_fn: Callable,
                    device: str,
                    uda_loader: Optional[DataLoader] = None,
                    eval_loaders: Optional[Mapping[str, DataLoader]] = None,
                    max_examples_per_epoch: Optional[int] = None
                    ) -> float:
        """Conduct one epoch of training and return the loss."""

        loaders = [x for x in train_loaders.values()]
        train_minibatches_iterator = zip(*loaders)

        def _prepare_batch(batch) -> Tuple[Tensor, Tensor]:
            x_batch, y_batch, _, _ = unpack_batch(batch)
            return x_batch.float().to(device), y_batch.float().to(device)

        loss = None
        examples_seen = 0
        while True:
            print(f"{self.__class__.__name__}:train examples seen: "
                  f"{examples_seen} of {max_examples_per_epoch}")
            minibatches_device = [_prepare_batch(batch) for batch in
                                  next(train_minibatches_iterator)]
            # Note: if this was a domain_adaption task, do the same as above
            # for uda_loader.
            tmp = self.update(minibatches_device)

            loss = tmp['loss'] if loss is None else loss + tmp['loss']
            examples_seen += sum(len(x) for x in minibatches_device)
            if examples_seen >= max_examples_per_epoch:
                break

        return loss / examples_seen

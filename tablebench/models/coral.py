from typing import Mapping, Optional, Callable

import torch
from torch.nn.functional import binary_cross_entropy_with_logits
from tablebench.models.rtdl import MLPModel
from tablebench.models.torchutils import unpack_batch, apply_model
from tablebench.models.losses import CORALLoss


def domain_generalization_train_epoch(
        model: MLPModel, optimizer, criterion: CORALLoss,
        id_train_loader: torch.utils.data.DataLoader,
        ood_train_loader: torch.utils.data.DataLoader,
        device):
    model.train()
    running_loss = 0.0
    n_train = 0
    batch_size = id_train_loader.batch_size

    block_num = criterion.block_num
    layer = criterion.layer
    # The key used to find the activations in the dictionary.
    activations_key = f'block{block_num}{layer}'

    activation = {}

    def get_activation():
        """Utility function to fetch an activation."""

        def hook(model, input, output):
            activation[activations_key] = output.detach()

        return hook

    model.blocks[block_num].linear.register_forward_hook(get_activation())

    for id_batch, ood_batch in zip(id_train_loader, ood_train_loader):
        inputs_id, labels_id, _, _ = unpack_batch(id_batch)
        inputs_ood, _, _, _ = unpack_batch(ood_batch)

        if len(inputs_id) != len(inputs_ood):
            print(f"Inconsistent batch sizes for CORAL loss ({len(inputs_id)}"
                  f"vs {len(inputs_ood)}; this can occur in the final batch."
                  f"Skipping.")
            continue

        inputs_id.float().to(device)
        labels_id.float().to(device)
        inputs_ood.float().to(device)

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs_id = apply_model(model, inputs_id).squeeze()
        activations_id = activation[activations_key]
        _ = apply_model(model, inputs_ood).squeeze()
        activations_ood = activation[activations_key]

        # Normalize CORAL loss by the batch size, so its scale is
        # batch size-independent.
        coral_loss = criterion(activations_id, activations_ood) / batch_size
        ce_loss = binary_cross_entropy_with_logits(input=outputs_id,
                                                   target=labels_id)
        loss = ce_loss + criterion.loss_lambda * coral_loss

        loss.backward()
        optimizer.step()

        # print statistics
        running_loss += loss.item()
        n_train += len(inputs_id)

    return running_loss / n_train


# TODO(jpgard): implement this in a way that takes a generic class, or maybe
#  a function that produces a model?
class DeepCoralModel(MLPModel):

    def train_epoch(self, train_loader: torch.utils.data.DataLoader,
                    optimizer: torch.optim.Optimizer,
                    loss_fn: Callable,
                    device: str,
                    other_loaders: Optional[
                        Mapping[str, torch.utils.data.DataLoader]] = None,
                    ood_loader_key="ood_test",
                    ):
        """Run a single epoch of model training."""

        domain_generalization_train_epoch(self, optimizer, loss_fn, train_loader,
                                          other_loaders[ood_loader_key], device)
        # train_epoch(self, optimizer, loss_fn, train_loader, device=device)

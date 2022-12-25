from fairlearn.reductions import ErrorRateParity
from frozendict import frozendict

from tablebench.core import TabularDataset
from tablebench.models.compat import is_pytorch_model_name
from tablebench.models.losses import GroupDROLoss, CORALLoss
from torch.nn import functional as F

_DEFAULT_CONFIGS = frozendict({
    "deepcoral":
        {"num_layers": 4,
         "d_hidden": 512,
         "dropouts": 0.},
    "expgrad":
        {"constraints": ErrorRateParity()},
    "ft_transformer":
        {"cat_cardinalities": None,
         "n_blocks": 1},
    "group_dro":
        {"num_layers": 2,
         "d_hidden": 256,
         "group_weights_step_size": 0.05},
    "mlp":
        {"num_layers": 2,
         "d_hidden": 256,
         "dropouts": 0.},
    "resnet":
        {"n_blocks": 2,
         "dropout_first": 0.2,
         "dropout_second": 0.,
         "d_main": 128,
         "d_hidden": 256},

})


def get_default_config(model: str, dset: TabularDataset) -> dict:
    """Get a default config for a model by name."""
    config = _DEFAULT_CONFIGS.get(model, {})

    if is_pytorch_model_name(model) and model != "ft_transformer":
        config.update({"d_in": dset.X_shape[1],
                       "activation": "ReLU"})
    elif is_pytorch_model_name(model):
        config.update({"n_num_features": dset.X_shape[1]})

    if model == "group_dro":
        config["n_groups"] = dset.n_domains
        config["criterion"] = GroupDROLoss(n_groups=dset.n_domains)

    elif model == "deepcoral":
        config["criterion"] = CORALLoss()

    else:
        config["criterion"] = F.binary_cross_entropy_with_logits

    if is_pytorch_model_name(model):
        config.update({"batch_size": 64,
                       "lr": 0.01,
                       "weight_decay": 0.01,
                       "n_epochs": 1})

    return config

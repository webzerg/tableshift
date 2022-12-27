from fairlearn.reductions import ErrorRateParity
from frozendict import frozendict

from tablebench.core import TabularDataset
from tablebench.models.compat import is_pytorch_model_name
from tablebench.models.losses import GroupDROLoss, CORALLoss, DROLoss
from torch.nn import functional as F

# Default configs for testing models. These are not tuned
# or selected for any particular reason; they might not even
# be good choices for hyperparameters.

_DEFAULT_CONFIGS = frozendict({
    "deepcoral":
        {"num_layers": 4,
         "d_hidden": 512,
         "loss_lambda": 0.01,
         "dropouts": 0.},
    "dro":
        {"num_layers": 2,
         "d_hidden": 512,
         "dropouts": 0.,
         "geometry": "cvar",
         "size": 0.5,
         "reg": 0.01,
         "max_iter": 1000},
    "expgrad":
        {"constraints": ErrorRateParity()},
    "ft_transformer":
        {"cat_cardinalities": None,
         "n_blocks": 1,
         "residual_dropout": 0.,
         "attention_dropout": 0.,
         "ffn_dropout": 0.,
         "ffn_factor": 1.,
         # This is feature embedding size in Table 13 above.
         "d_token": 64,
         },
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

    # Models that use non-cross-entropy training objectives.
    if model == "deepcoral":
        config["criterion"] = CORALLoss()

    elif model == "dro":
        config["criterion"] = DROLoss(size=config["size"],
                                      reg=config["reg"],
                                      geometry=config["geometry"],
                                      max_iter=config["max_iter"])
    elif model == "group_dro":
        config["n_groups"] = dset.n_domains
        config["criterion"] = GroupDROLoss(n_groups=dset.n_domains)


    else:
        config["criterion"] = F.binary_cross_entropy_with_logits

    if is_pytorch_model_name(model):
        config.update({"batch_size": 64,
                       "lr": 0.01,
                       "weight_decay": 0.01,
                       "n_epochs": 1})

    return config

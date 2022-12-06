from fairlearn.reductions import ErrorRateParity
from frozendict import frozendict

from tablebench.core import TabularDataset
from tablebench.models.compat import is_pytorch_model_name
from tablebench.models.dro import group_dro_loss, GroupDROModel
from torch.nn import functional as F

# TODO(jpgard): set all architectural defaults here
#  based on [gorishniy2021revisiting] paper.
_DEFAULT_CONFIGS = frozendict({
    "expgrad": {"constraints": ErrorRateParity()},
    "ft_transformer": dict(cat_cardinalities=None),
    "group_dro": dict(d_layers=[256, 256],
                      group_weights_step_size=0.05),
    "mlp": dict(d_layers=[256, 256]),
    "resnet": dict(),
})


def get_model_config(model: str, dset: TabularDataset) -> dict:
    """Get a default config for a model by name."""
    config = _DEFAULT_CONFIGS.get(model, {})

    if is_pytorch_model_name(model) and model != "ft_transformer":
        config.update({"d_in": dset.X_shape[1]})
    elif is_pytorch_model_name(model):
        config.update({"n_num_features": dset.X_shape[1]})

    if model == "group_dro":
        config["n_groups"] = dset.n_domains
        config["criterion"] = group_dro_loss

    else:
        config["criterion"] = F.binary_cross_entropy_with_logits

    if is_pytorch_model_name(model):
        config.update({"batch_size": 512,
                       "n_epochs": 1})

    return config

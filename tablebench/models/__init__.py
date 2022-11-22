from lightgbm import LGBMClassifier
from sklearn.ensemble import HistGradientBoostingClassifier
import xgboost as xgb

from tablebench.models.expgrad import ExponentiatedGradient
from tablebench.models.rtdl import ResNetModel, MLPModel, FTTransformerModel
from tablebench.models.wcs import WeightedCovariateShiftClassifier
from tablebench.models.dro import GroupDROModel

SKLEARN_MODELS = ("expgrad",
                  "histgbm",
                  "lightgbm",
                  "wcs",
                  "xgb")
PYTORCH_MODELS = ("ft_transformer",
                  "group_dro",
                  "mlp",
                  "resnet")

_DEFAULT_CONFIGS = {
    "mlp": dict(d_layers=[256, 256]),
    "ft_transformer": dict(cat_cardinalities=None),
    "resnet": dict(),
    "group_dro": dict(d_layers=[256, 256],
                      group_weights_step_size=0.05),

}


def get_pytorch_model_config(model, dset) -> dict:
    """Get a default config for a pytorch model."""
    config = _DEFAULT_CONFIGS[model]
    config.update({"d_in": dset.X_shape[1]})
    if model == "group_dro":
        config["n_groups"] = dset.n_groups
    return config


def get_estimator(model, d_out=1, **kwargs):
    if model == "expgrad":
        return ExponentiatedGradient(**kwargs)
    elif model == "ft_transformer":
        tconfig = FTTransformerModel.get_default_transformer_config()
        tconfig["last_layer_query_idx"] = [-1]
        tconfig["d_out"] = 1
        return FTTransformerModel._make(**kwargs, transformer_config=tconfig)
    elif model == "group_dro":
        return GroupDROModel(d_out=d_out, dropouts=0., activation='ReLU',
                             **kwargs)
    elif model == "histgbm":
        return HistGradientBoostingClassifier(**kwargs)
    elif model == "lightgbm":
        return LGBMClassifier(**kwargs)
    elif model == "mlp":
        return MLPModel(d_out=d_out, dropouts=0., activation='ReLU', **kwargs)
    elif model == "resnet":
        return ResNetModel(
            n_blocks=2,
            d_main=128,
            d_hidden=256,
            dropout_first=0.2,
            dropout_second=0.0,
            normalization='BatchNorm1d',
            activation='ReLU',
            d_out=d_out,
            **kwargs
        )
    elif model == "wcs":
        # Weighted Covariate Shift classifier.
        return WeightedCovariateShiftClassifier()
    elif model == "xgb":
        return xgb.XGBClassifier(**kwargs)
    else:
        raise NotImplementedError(f"model {model} not implemented.")

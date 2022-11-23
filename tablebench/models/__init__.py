from typing import Any

from fairlearn.reductions import ErrorRateParity
from frozendict import frozendict
from lightgbm import LGBMClassifier
from sklearn.ensemble import HistGradientBoostingClassifier
import xgboost as xgb

from tablebench.core import TabularDataset, DomainSplitter
from tablebench.models.expgrad import ExponentiatedGradient
from tablebench.models.rtdl import ResNetModel, MLPModel, FTTransformerModel
from tablebench.models.wcs import WeightedCovariateShiftClassifier
from tablebench.models.dro import GroupDROModel

SKLEARN_MODEL_CLS = {"expgrad": ExponentiatedGradient,
                     "histgbm": HistGradientBoostingClassifier,
                     "lightgbm": LGBMClassifier,
                     "wcs": WeightedCovariateShiftClassifier,
                     "xgb": xgb.XGBClassifier}
PYTORCH_MODEL_CLS = {"ft_transformer": FTTransformerModel,
                     "group_dro": GroupDROModel,
                     "mlp": MLPModel,
                     "resnet": ResNetModel}

# TODO(jpgard): set all architectural defaults here
#  based on [gorishniy2021revisiting] paper.
_DEFAULT_CONFIGS = frozendict({
    "ft_transformer": dict(cat_cardinalities=None),
    "group_dro": dict(d_layers=[256, 256],
                      group_weights_step_size=0.05),
    "mlp": dict(d_layers=[256, 256]),
    "resnet": dict(),
})


def is_pytorch_model(model: Any) -> bool:
    """Helper function to determine whether a model object is a pytorch model.

    If True, uses the standard pytorch training loop defined in
    tableshift.models.training.train_pytorch(); if False,
    uses the sklearn training loop defined in
    tableshift.models.training.train_sklearn()."""
    is_sklearn = isinstance(model, tuple(SKLEARN_MODEL_CLS.values()))
    is_pt = isinstance(model, tuple(PYTORCH_MODEL_CLS.values()))
    assert is_sklearn or is_pt, f"unknown model type {type(model)}"
    return is_pt


def is_pytorch_model_name(model: str) -> bool:
    """Helper function to determine whether a model name is a pytorch model.

    ISee description of is_pytorch_model() above."""
    is_sklearn = model in SKLEARN_MODEL_CLS
    is_pt = model in PYTORCH_MODEL_CLS
    assert is_sklearn or is_pt, f"unknown model name {model}"
    return is_pt


def get_model_config(model: str, dset: TabularDataset) -> dict:
    """Get a default config for a model by name."""
    config = _DEFAULT_CONFIGS.get(model, {})

    if is_pytorch_model_name(model) and model != "ft_transformer":
        config.update({"d_in": dset.X_shape[1]})
    elif is_pytorch_model_name(model):
        config.update({"n_num_features": dset.X_shape[1]})

    if model == "group_dro":
        config["n_groups"] = dset.n_domains

    if is_pytorch_model_name(model):
        config.update({"batch_size": 512})

    if model == "expgrad":
        assert isinstance(dset.splitter, DomainSplitter)
        config.update(
            {"domain_feature_colname": [dset.splitter.domain_split_varname],
             "estimator": xgb.XGBClassifier(),
             "constraints": ErrorRateParity()})
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
        return MLPModel(d_in=kwargs["d_in"],
                        d_layers=kwargs["d_layers"],
                        d_out=d_out,
                        dropouts=0.,
                        activation='ReLU',
                        )
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

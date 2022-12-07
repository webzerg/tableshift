import xgboost as xgb
from lightgbm import LGBMClassifier
from sklearn.ensemble import HistGradientBoostingClassifier

from tablebench.models.expgrad import ExponentiatedGradient
from tablebench.models.rtdl import ResNetModel, MLPModel, FTTransformerModel
from tablebench.models.wcs import WeightedCovariateShiftClassifier
from tablebench.models.dro import GroupDROModel


def get_estimator(model, d_out=1, **kwargs):
    if model == "expgrad":
        return ExponentiatedGradient(**kwargs)
    elif model == "ft_transformer":
        tconfig = FTTransformerModel.get_default_transformer_config()
        tconfig["last_layer_query_idx"] = [-1]
        tconfig["d_out"] = 1
        return FTTransformerModel._make(
            n_num_features=kwargs["n_num_features"],
            cat_cardinalities=kwargs["cat_cardinalities"],
            transformer_config=tconfig)
    elif model == "group_dro":
        return GroupDROModel(
            d_in=kwargs["d_in"],
            d_layers=[kwargs["d_hidden"]] * kwargs["num_layers"],
            d_out=d_out,
            dropouts=0.,
            activation='ReLU',
            group_weights_step_size=kwargs["group_weights_step_size"],
            n_groups=kwargs["n_groups"])
    elif model == "histgbm":
        return HistGradientBoostingClassifier(**kwargs)
    elif model == "lightgbm":
        return LGBMClassifier(**kwargs)
    elif model == "mlp":
        return MLPModel(d_in=kwargs["d_in"],
                        d_layers=[kwargs["d_hidden"]] * kwargs["num_layers"],
                        d_out=d_out,
                        dropouts=0.,
                        activation='ReLU',
                        )
    elif model == "resnet":
        return ResNetModel(
            d_in=kwargs["d_in"],
            n_blocks=2,
            d_main=128,
            d_hidden=256,
            dropout_first=0.2,
            dropout_second=0.0,
            normalization='BatchNorm1d',
            activation='ReLU',
            d_out=d_out)

    elif model == "wcs":
        # Weighted Covariate Shift classifier.
        return WeightedCovariateShiftClassifier(**kwargs)
    elif model == "xgb":
        return xgb.XGBClassifier(**kwargs)
    else:
        raise NotImplementedError(f"model {model} not implemented.")

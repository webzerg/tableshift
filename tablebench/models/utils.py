import copy

import xgboost as xgb
from lightgbm import LGBMClassifier
from sklearn.ensemble import HistGradientBoostingClassifier

from tablebench.models.compat import OPTIMIZER_ARGS
from tablebench.models.coral import DeepCoralModel, MMDModel
from tablebench.models.dro import GroupDROModel
from tablebench.models.expgrad import ExponentiatedGradient
from tablebench.models.irm import IRMModel
from tablebench.models.rtdl import ResNetModel, MLPModel, FTTransformerModel
from tablebench.models.wcs import WeightedCovariateShiftClassifier
from tablebench.models.mixup import MixUpModel


def get_estimator(model, d_out=1, **kwargs):
    if model == "deepcoral":
        return DeepCoralModel(d_in=kwargs["d_in"],
                              d_layers=[kwargs["d_hidden"]] * kwargs[
                                  "num_layers"],
                              d_out=d_out,
                              dropouts=kwargs["dropouts"],
                              activation=kwargs["activation"],
                              mmd_gamma=kwargs["mmd_gamma"],
                              **{k: kwargs[k] for k in OPTIMIZER_ARGS})
    elif model == "expgrad":
        return ExponentiatedGradient(**kwargs)
    elif model == "ft_transformer":
        tconfig = FTTransformerModel.get_default_transformer_config()

        tconfig["last_layer_query_idx"] = [-1]
        tconfig["d_out"] = 1
        params_to_override = ("n_blocks", "residual_dropout", "d_token",
                              "attention_dropout", "ffn_dropout")
        for k in params_to_override:
            tconfig[k] = kwargs[k]

        tconfig["ffn_d_hidden"] = int(kwargs["d_token"] * kwargs["ffn_factor"])

        # Fixed as in https://arxiv.org/pdf/2106.11959.pdf
        tconfig['attention_n_heads'] = 8

        # Hacky way to construct a FTTransformer model
        model = FTTransformerModel._make(
            n_num_features=kwargs["n_num_features"],
            cat_cardinalities=kwargs["cat_cardinalities"],
            transformer_config=tconfig)
        tconfig.update({k: kwargs[k] for k in OPTIMIZER_ARGS})
        model.config = copy.deepcopy(tconfig)
        model._init_optimizer()

        return model

    elif model == "group_dro":
        return GroupDROModel(
            d_in=kwargs["d_in"],
            d_layers=[kwargs["d_hidden"]] * kwargs["num_layers"],
            d_out=d_out,
            dropouts=kwargs["dropouts"],
            activation=kwargs["activation"],
            group_weights_step_size=kwargs["group_weights_step_size"],
            n_groups=kwargs["n_groups"],
            **{k: kwargs[k] for k in OPTIMIZER_ARGS},
        )
    elif model == "histgbm":
        return HistGradientBoostingClassifier(**kwargs)

    elif model == "irm":
        return IRMModel(
            d_in=kwargs["d_in"],
            d_layers=[kwargs["d_hidden"]] * kwargs["num_layers"],
            d_out=d_out,
            dropouts=kwargs["dropouts"],
            activation=kwargs["activation"],
            irm_lambda=kwargs['irm_lambda'],
            irm_penalty_anneal_iters=kwargs['irm_penalty_anneal_iters'],
            **{k: kwargs[k] for k in OPTIMIZER_ARGS}, )

    elif model == "lightgbm":
        return LGBMClassifier(**kwargs)

    elif model == "mixup":
        return MixUpModel(
            d_in=kwargs["d_in"],
            d_layers=[kwargs["d_hidden"]] * kwargs["num_layers"],
            d_out=d_out,
            dropouts=kwargs["dropouts"],
            activation=kwargs["activation"],
            mixup_alpha=kwargs["mixup_alpha"],
            **{k: kwargs[k] for k in OPTIMIZER_ARGS}
        )

    elif model == "mlp" or model == "dro":
        return MLPModel(d_in=kwargs["d_in"],
                        d_layers=[kwargs["d_hidden"]] * kwargs["num_layers"],
                        d_out=d_out,
                        dropouts=kwargs["dropouts"],
                        activation=kwargs["activation"],
                        **{k: kwargs[k] for k in OPTIMIZER_ARGS})

    elif model == "mmd":
        return MMDModel(d_in=kwargs["d_in"],
                        d_layers=[kwargs["d_hidden"]] * kwargs[
                            "num_layers"],
                        d_out=d_out,
                        dropouts=kwargs["dropouts"],
                        activation=kwargs["activation"],
                        mmd_gamma=kwargs["mmd_gamma"],
                        **{k: kwargs[k] for k in OPTIMIZER_ARGS})
    elif model == "resnet":
        d_hidden = kwargs["d_main"] * kwargs["hidden_factor"]
        return ResNetModel(
            d_in=kwargs["d_in"],
            n_blocks=kwargs["n_blocks"],
            d_main=kwargs["d_main"],
            d_hidden=d_hidden,
            dropout_first=kwargs["dropout_first"],
            dropout_second=kwargs["dropout_second"],
            normalization='BatchNorm1d',
            activation=kwargs["activation"],
            d_out=d_out,
            **{k: kwargs[k] for k in OPTIMIZER_ARGS},
        )

    elif model == "wcs":
        # Weighted Covariate Shift classifier.
        return WeightedCovariateShiftClassifier(**kwargs)
    elif model == "xgb":
        return xgb.XGBClassifier(**kwargs)
    else:
        raise NotImplementedError(f"model {model} not implemented.")

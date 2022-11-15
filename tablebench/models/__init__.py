from lightgbm import LGBMClassifier
from sklearn.ensemble import HistGradientBoostingClassifier
import xgboost as xgb

from tablebench.models.rtdl import ResNetModel


def get_estimator(model, **kwargs):
    if model == "histgbm":
        return HistGradientBoostingClassifier(**kwargs)
    elif model == "lightgbm":
        return LGBMClassifier(**kwargs)
    elif model == "resnet":
        assert "d_in" in kwargs, "missing required argument d_in."
        return ResNetModel.make_baseline(
            n_blocks=2,
            d_main=128,
            d_hidden=256,
            dropout_first=0.2,
            dropout_second=0.0,
            # must be num_classes if multiclass (non-binary) problem.
            d_out=1,
            **kwargs
        )
    elif model == "xgb":
        return xgb.XGBClassifier(**kwargs)
    else:
        raise NotImplementedError(f"model {model} not implemented.")

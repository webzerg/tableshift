from lightgbm import LGBMClassifier
from sklearn.ensemble import HistGradientBoostingClassifier
import xgboost as xgb

from tablebench.models.rtdl import ResNetModel, MLPModel, FTTransformerModel
from tablebench.models.wcs import WeightedCovariateShiftClassifier

def get_estimator(model, **kwargs):
    if model == "ft_transformer":
        return FTTransformerModel.make_baseline(last_layer_query_idx=[-1],
                                               d_out=1, **kwargs)
    elif model == "histgbm":
        return HistGradientBoostingClassifier(**kwargs)
    elif model == "lightgbm":
        return LGBMClassifier(**kwargs)
    elif model == "mlp":
        return MLPModel.make_baseline(d_out=1, dropout=0., **kwargs)
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
    elif model == "wcs":
        # Weighted Covariate Shift classifier.
        return WeightedCovariateShiftClassifier()
    elif model == "xgb":
        return xgb.XGBClassifier(**kwargs)
    else:
        raise NotImplementedError(f"model {model} not implemented.")

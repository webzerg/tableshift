import xgboost as xgb
from sklearn.ensemble import HistGradientBoostingClassifier


def get_estimator(model):
    if model == "histgbm":
        return HistGradientBoostingClassifier()
    elif model == "xgb":
        return xgb.XGBClassifier()

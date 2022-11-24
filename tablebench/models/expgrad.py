import warnings

import fairlearn.reductions
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import xgboost as xgb

assert fairlearn.__version__.split('.')[1] == '7'


class ExponentiatedGradient(fairlearn.reductions.ExponentiatedGradient):
    """Custom class to allow for scikit-learn-compatible interface.

    Specifically, this method takes (and ignores) a sample_weights
    parameter to its .fit() method; otherwise identical to
    fairlearn.ExponentiatedGradient.
    """

    def __init__(self, constraints, eps: float = 0.01, eta0: float = 2.,
                 base_estimator_cls=xgb.XGBClassifier,
                 **kwargs):
        estimator = base_estimator_cls(**kwargs)
        super().__init__(estimator=estimator,
                         constraints=constraints,
                         eps=eps,
                         eta0=eta0)

        # The LabelEncoder is used to ensure sensitive features are of
        # numerical type (not string/categorical).
        self.le = LabelEncoder()

    def fit(self, X: pd.DataFrame, y: pd.Series,
            sample_weight=None, **kwargs):
        del sample_weight

        # Fetch and check the domain labels
        assert "d" in kwargs, "require 'd', array with domain labels."
        d = kwargs.pop("d")
        assert isinstance(d, pd.Series)

        # Numerically encode the sensitive attribute; fairlearn.reductions
        # does not accept categorical/string-type data.
        domains_enc = self.le.fit_transform(d.values)

        with warnings.catch_warnings():
            # Filter FutureWarnings raised by fairlearn that overwhelm the
            # console output.
            warnings.filterwarnings("ignore", category=FutureWarning)

            super().fit(X.values, y.values, sensitive_features=domains_enc,
                        **kwargs)

    def predict(self, X, random_state=None):
        return super().predict(X)

    def predict_proba(self, X):
        """Alias to _pmf_predict(). Note that this tends to return very
        close to 'hard' predictions (probabilities close to 0/1), which don't
        perform well for metrics like cross-entropy."""
        return super()._pmf_predict(X)

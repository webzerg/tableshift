from typing import List

import fairlearn.reductions
from sklearn.preprocessing import LabelEncoder

assert fairlearn.__version__.split('.')[1] == '7'


class ExponentiatedGradient(fairlearn.reductions.ExponentiatedGradient):
    """Custom class to allow for scikit-learn-compatible interface.

    Specifically, this method takes (and ignores) a sample_weights
    parameter to its .fit() method; otherwise identical to
    fairlearn.ExponentiatedGradient.
    """

    def __init__(self, sensitive_feature_colnames: List[str], **kwargs):
        super().__init__(**kwargs)
        # We save the names of the sensitive features as a class attribute
        # so that they are not required at train time; this provides a
        # consistent sklearn-style interface for the model.
        self.sensitive_feature_colnames = sensitive_feature_colnames
        # The LabelEncoder is used to ensure sensitive features are of
        # numerical type (not string/categorical).
        self.le = LabelEncoder()

    def _prepare_x(self, X):
        """Helper function to drop sensitive columns from data matrix."""
        return X.drop(columns=self.sensitive_feature_colnames)

    def fit(self, X, y, sample_weight=None, **kwargs):
        del sample_weight

        # Numerically encode the sensitive attribute; fairlearn.reductions
        # does not accept categorical/string-type data.
        sens = self.le.fit_transform(X[self.sensitive_feature_colnames].values)

        X_ = self._prepare_x(X)
        super().fit(
            X_.values, y.values,
            sensitive_features=sens,
            **kwargs)

    def predict(self, X, random_state=None):
        return super().predict(self._prepare_x(X))

    def predict_proba(self, X):
        """Alias to _pmf_predict(). Note that this tends to return very
        close to 'hard' predictions (probabilities close to 0/1), which don't
        perform well for metrics like cross-entropy."""
        return super()._pmf_predict(self._prepare_x(X))

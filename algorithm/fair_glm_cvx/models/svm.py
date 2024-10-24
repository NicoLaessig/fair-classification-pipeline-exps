
'''
    Linear SVM
'''

import numpy as np
from sklearn.svm import NuSVC
from . import BaseFairEstimator


class LinearSVM(BaseFairEstimator):
    def __init__(
            self,
            sensitive_index=0,
            sensitive_predictor=False,
            standardize=False,
            nu=0.1):

        self.nu = nu
        self._base_clf = None

        super().__init__(
            sensitive_index=sensitive_index,
            sensitive_predictor=sensitive_predictor,
            standardize=standardize
        )

    def fit(self, X, y):
        y = y.astype(np.float64)

        assert type(y) == np.ndarray
        assert type(X) == np.ndarray
        assert len(X) == len(y)

        X, A = self._process_predictors(X)

        self._base_clf = NuSVC(probability=True, nu=self.nu, kernel='linear')
        self._base_clf.fit(X, y)

    def _predict(self, X):
        pred = self._base_clf.predict(X)
        return pred

    def _predict_proba(self, X):
        pred = self._base_clf.predict_proba(X)
        return pred

    def decision_function(self, X):
        pred = self._base_clf.decision_function(X)
        return pred


'''
    Implementation of the Linear Fair ERM method proposed in
    Donini, M., Oneto, L., Ben-David, S., Shawe-Taylor, J. S., & Pontil, M. (2018). Empirical risk minimization under fairness constraints. Advances in Neural Information Processing Systems, 31.
    Original implementation of this method can be found on https://github.com/jmikko/fair_ERM
'''

import sys
import numpy as np
from sklearn.svm import NuSVC
from . import BaseFairEstimator


class LinearFERM(BaseFairEstimator):
    def __init__(
            self,
            sensitive_index=0,
            sensitive_predictor=False,
            standardize=False,
            nu=0.1):

        self.nu = nu
        self._base_clf = None

        self.u = None
        self.max_i = None

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

        val0 = np.min(A)
        val1 = np.max(A)
        
        # Evaluation of the empirical averages among the groups
        tmp = [ex for idx, ex in enumerate(X)
               if y[idx] == 1 and A[idx] == val1]
        average_A_1 = np.mean(tmp, 0)
        tmp = [ex for idx, ex in enumerate(X)
               if y[idx] == 1 and A[idx] == val0]
        average_not_A_1 = np.mean(tmp, 0)

        # Evaluation of the vector u (difference among the two averages)
        self.u = -(average_A_1 - average_not_A_1)
        self.max_i = np.argmax(self.u)
        
        # Application of the new representation
        newX = np.array([ex - self.u * (ex[self.max_i] / self.u[self.max_i]) for ex in X])
        newX = np.delete(newX, self.max_i, 1)
        
        # Fitting the linear model by using the new data
        self._base_clf = NuSVC(probability=True, nu=self.nu, kernel='linear')
        self._base_clf.fit(newX, y)

    def new_representation(self, X):
        if self.u is None:
            sys.exit('Model not trained yet!')
            return 0

        X, A = self._process_predictors(X)

        newX = np.array([ex - self.u * (ex[self.max_i] / self.u[self.max_i]) for ex in X])
        newX = np.delete(newX, self.max_i, 1)
        return newX

    def _predict(self, X):
        X = self.new_representation(X)
        pred = self._base_clf.predict(X)
        return pred

    def _predict_proba(self, X):
        X = self.new_representation(X)
        pred = self._base_clf.predict_proba(X)
        return pred

    def decision_function(self, X):
        X = self.new_representation(X)
        pred = self._base_clf.decision_function(X)
        return pred

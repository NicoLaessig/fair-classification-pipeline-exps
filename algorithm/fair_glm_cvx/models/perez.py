
'''
    Implementation of the HSIC Penalized Fair Linear Regression proposed in
    Pérez-Suay, A., Laparra, V., Mateo-García, G., Muñoz-Marí, J., Gómez-Chova, L., & Camps-Valls, G. (2017, September). Fair kernel learning. In Joint European Conference on Machine Learning and Knowledge Discovery in Databases (pp. 339-355). Springer, Cham.
'''

import numpy as np
from . import BaseFairEstimator


class HSICLinearRegression(BaseFairEstimator):
    def __init__(
            self,
            sensitive_index=0,
            sensitive_predictor=False,
            standardize=False,
            lam=0.1,
            fit_intercept=True,
            maxiter=100,
            tol=1e-4
            ):

        super().__init__(
            sensitive_index=sensitive_index,
            sensitive_predictor=sensitive_predictor,
            standardize=standardize
        )

        self.lam = lam
        self.fit_intercept = fit_intercept
        self.maxiter = maxiter
        self.tol = tol
        self.coef_ = None

    def fit(self, X, y):

        y = y.astype(np.float64)

        assert type(y) == np.ndarray
        assert type(X) == np.ndarray
        assert len(X) == len(y)

        X, A = self._process_predictors(X)

        if self.fit_intercept:
            X = np.column_stack([np.ones(len(X)), X])

        n, p = X.shape

        A = A.reshape(-1, 1)

        beta = np.linalg.inv(X.T.dot(X) + X.T.dot(A).dot(A.T).dot(X)*self.lam/n).dot(X.T.dot(y))
        self.coef_ = beta.flatten()

    def _predict(self, X):
        X, A = self._process_predictors(X)
        
        if self.fit_intercept:
            X = np.column_stack([np.ones(len(X)), X])

        return np.dot(X, self.coef_)

    def _predict_proba(self, X):
        raise NotImplementedError

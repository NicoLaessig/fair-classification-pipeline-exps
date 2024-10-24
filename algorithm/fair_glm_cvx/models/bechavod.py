
'''
    Implementation of the Squared Difference Penalizer proposed in
    Bechavod, Y., & Ligett, K. (2017). Penalizing unfairness in binary classification. arXiv preprint arXiv:1707.00044.
'''

import numpy as np
from itertools import combinations
from . import BaseFairEstimator

class SquaredDifferenceFairLogistic(BaseFairEstimator):
    def __init__(
            self,
            sensitive_index=0,
            sensitive_predictor=False,
            standardize=False,
            lam=0.,
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
        self.coef_traj = None

    def fit(self, X, y):

        y = y.astype(np.float64)

        assert type(y) == np.ndarray
        assert type(X) == np.ndarray
        assert len(X) == len(y)

        X, A = self._process_predictors(X)

        if self.fit_intercept:
            X = np.column_stack([np.ones(len(X)), X])

        n, p = X.shape

        D = []
        for y_ in set(y):
            for a, b in combinations(set(A), 2):
                D.append((X[np.logical_and(A == a, y == y_)].mean(0) - X[np.logical_and(A == b, y == y_)].mean(0)))

        D = np.row_stack(D)
        D = self.lam*D.T.dot(D)

        beta = [np.zeros(p)]
        for i in range(self.maxiter):
            mu = 1 / (1 + np.exp(-np.dot(X, beta[i])))
            grad = -X.T.dot(y-mu) / n + D.dot(beta[i])

            w = np.diag(mu * (1 - mu))
            hinv = np.linalg.inv(X.T.dot(w).dot(X)/n + D)

            beta.append(beta[i] - np.dot(hinv, grad))

            if np.linalg.norm(beta[i+1] - beta[i]) < self.tol:
                break

        self.coef_traj = beta
        self.coef_ = beta[-1]

    def _predict(self, X):
        X, A = self._process_predictors(X)
        if self.fit_intercept:
            X = np.column_stack([np.ones(len(X)), X])
        p = 1 / (1 + np.exp(-np.dot(X, self.coef_).flatten()))
        return (p > 0.5)*1
        
    def _predict_proba(self, X):
        if self.fit_intercept:
            X = np.column_stack([np.ones(len(X)), X])
        p = 1 / (1 + np.exp(-np.dot(X, self.coef_).flatten()))
        return np.column_stack([1 - p, p])

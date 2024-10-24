
'''
    Implementation of the Individual and Group Fairness Methods proposed in
    Berk, R., Heidari, H., Jabbari, S., Joseph, M., Kearns, M., Morgenstern, J., ... & Roth, A. (2017). A convex framework for fair regression. arXiv preprint arXiv:1706.02409.
'''

import numpy as np
from . import BaseFairEstimator
from itertools import combinations
from sklearn.utils.multiclass import unique_labels


class ConvexFrameworkModel(BaseFairEstimator):
    def __init__(
            self,
            sensitive_index=0,
            sensitive_predictor=False,
            standardize=False,
            family='bernoulli',
            penalty='individual',
            lam=0.,
            fit_intercept=True,
            maxiter=100,
            tol=1e-4):

        assert family in ['bernoulli', 'normal'], f'Family {family} is not supported. Choose one of [\'bernoulli\', \'normal\'].'
        assert penalty in ['individual', 'group'], f'Penalty {penalty} is not supported. Choose one of [\'individual\', \'group\'].'

        self.lam = lam
        self.maxiter = maxiter
        self.fit_intercept = fit_intercept
        self.tol = tol
        self.family = family
        self.penalty = penalty
        self.coef_ = None
        self.coef_traj = None

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

        if self.fit_intercept:
            X = np.column_stack([np.ones(len(X)), X])

        n, p = X.shape

        if self.family == 'bernoulli':
            # logistic regression
            dist = lambda y1, y2: np.float64(y1 == y2)
        elif self.family == 'normal':
            # linear regression
            dist = lambda y1, y2: np.exp(-np.square(y1-y2))
        else:
            raise Exception(f'Family {self.family} is not supported!')

        D = np.zeros((p, p))
        if self.penalty == 'individual':
            for a, b in combinations(set(A), 2):
                Xa, ya = X[A == a], y[A == a].reshape(-1, 1)
                Xb, yb = X[A == b], y[A == b].reshape(-1, 1)

                diff = dist(ya[None,:,:], yb[:,None,:])*(Xa[None,:,:] - Xb[:,None,:])
                diff = diff.reshape(-1, p)

                D += diff.T @ diff / (np.float64(len(Xa)) * np.float64(len(Xb)))

            D = self.lam * D

        elif self.penalty == 'group':
            for a, b in combinations(set(A), 2):
                Xa, ya = X[A == a], y[A == a].reshape(-1, 1)
                Xb, yb = X[A == b], y[A == b].reshape(-1, 1)
                diff = dist(ya[None,:,:], yb[:,None,:])*(Xa[None,:,:] - Xb[:,None,:])
                diff = diff.reshape(-1, p)
                val = np.sum(diff, 0, keepdims=True) / (np.float64(len(ya)) * np.float64(len(yb)))
                D += val.T @ val

            D = self.lam * D

        else:
            raise Exception(f'Fairness penalty {self.penalty} is not supported!')

        beta = [np.zeros(p)]
        if self.family == 'normal':
            H = X.T.dot(X) / n + D
            hinv = np.linalg.inv(H)

            for i in range(self.maxiter):
                mu = np.dot(X, beta[i])
                grad = -X.T.dot(y-mu) / n + np.dot(D, beta[i])

                beta.append(beta[i] - np.dot(hinv, grad))

                if np.linalg.norm(beta[i + 1] - beta[i]) < self.tol:
                    break

        elif self.family == 'bernoulli':
            self.classes_ = unique_labels(y)

            for i in range(self.maxiter):
                mu = 1 / (1 + np.exp(-np.clip(np.dot(X, beta[i]), -30, 30)))
                grad = -X.T.dot(y-mu) / n + np.dot(D, beta[i])

                w = np.diag(mu * (1 - mu))
                hinv = np.linalg.inv(X.T.dot(w).dot(X)/n + D)

                beta.append(beta[i] - np.dot(hinv, grad))

                if np.linalg.norm(beta[i + 1] - beta[i]) < self.tol:
                    break

        self.coef_traj = beta
        self.coef_ = beta[-1]

    def _predict(self, X):
        X, A = self._process_predictors(X)
        if self.fit_intercept:
            X = np.column_stack([np.ones(len(X)), X])

        if self.family == 'normal':
            return np.dot(X, self.coef_).flatten()
        elif self.family == 'bernoulli':
            p = 1 / (1 + np.exp(-np.dot(X, self.coef_).flatten()))
            return (p > 0.5) * 1

    def _predict_proba(self, X):
        if self.fit_intercept:
            X = np.column_stack([np.ones(len(X)), X])

        if self.family == 'normal':
            return np.dot(X, self.coef_).flatten()
        elif self.family == 'bernoulli':
            p = 1 / (1 + np.exp(-np.dot(X, self.coef_).flatten()))
            return np.column_stack([1 - p, p])

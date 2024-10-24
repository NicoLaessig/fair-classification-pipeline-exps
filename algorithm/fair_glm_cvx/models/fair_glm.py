
'''
    Implementation of the Fair GLM with Convex Penalty in
    Do, H., Putzel, P., Martin, A., Smyth, P., & Zhong, J. (2022, July) Fair Generalized Linear Models with a Convex Penalty, In International Conference on Machine Learning. PMLR
'''

import numpy as np

from time import time
from itertools import combinations, product
from sklearn.preprocessing import OneHotEncoder
from sklearn.utils.multiclass import unique_labels
from . import BaseFairEstimator
from ..util import discretization

M = 50

class FairGeneralizedLinearModel(BaseFairEstimator):
    def __init__(
            self,
            sensitive_index=0,
            sensitive_predictor=False,
            standardize=False,
            lam=0.,
            family='bernoulli',
            discretization='equal_count',
            max_segments=100,
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
        self.maxiter = maxiter
        self.fit_intercept = fit_intercept
        self.tol = tol
        self.family = family
        self.discretization = discretization
        self.max_segments = max_segments

        self.coef_ = None
        self.coef_traj = None
        self.classes_ = None

        supported_family = [
            'normal',
            'bernoulli',
            'multinomial',
            'poisson']

        self._discrete_family = [
            'bernoulli',
            'multinomial',
            'poisson']

        assert family in supported_family, f'{family} family is not supported'
        self._enc = OneHotEncoder(sparse=False)

    def fit(self, X, y):

        y = y.astype(np.float64)

        assert type(y) == np.ndarray
        assert type(X) == np.ndarray
        assert len(X) == len(y)

        X, A = self._process_predictors(X)

        if self.fit_intercept:
            X = np.column_stack([np.ones(len(X)), X])

        n, p = X.shape

        if self.family == 'multinomial':
            y = self._enc.fit_transform(y.reshape(-1, 1))

        YD = discretization(y, A,
                family=self.family,
                max_segments=self.max_segments,
                discretization=self.discretization)

        D_start = time()
        D = np.zeros((p, p))
        if self.lam > 0:
            for (a, b), yd in product(combinations(set(A), 2), set(YD)):
                Xay = X[np.logical_and(A == a, YD == yd)]
                Xby = X[np.logical_and(A == b, YD == yd)]
                diff = (Xay[None, :, :] - Xby[:, None, :]).reshape(-1, p)
                D += diff.T @ diff / (np.float64(len(Xay)) * np.float64(len(Xby)))

            D = (self.lam * 2) / (len(set(YD)) * len(set(A)) * (len(set(A)) - 1)) * D
        self.D_time = time() - D_start

        beta = [np.zeros(p)]
        time_traj = [0.]
        if self.family == 'normal':
            H = X.T.dot(X) / n + D
            hinv = np.linalg.inv(H)

            for i in range(self.maxiter):
                start_time = time()
                mu = np.dot(X, beta[i])
                grad = -X.T.dot(y - mu) / n + np.dot(D, beta[i])

                beta.append(beta[i] - np.dot(hinv, grad))

                time_traj.append(time() - start_time)
                if np.linalg.norm(grad) < self.tol:
                    break

        elif self.family == 'bernoulli':
            self.classes_ = unique_labels(y)

            for i in range(self.maxiter):
                start_time = time()
                mu = 1. / (1 + np.exp(np.clip(-np.dot(X, beta[i]), -M, M)))
                grad = -X.T.dot(y.reshape(-1,) - mu) / n + np.dot(D, beta[i])
                w = np.diag(mu * (1 - mu))
                hinv = np.linalg.inv(X.T.dot(w).dot(X) / n + D)

                beta.append(beta[i] - np.dot(hinv, grad))

                time_traj.append(time() - start_time)
                if np.linalg.norm(grad) < self.tol:
                    break

        elif self.family == 'poisson':
            for i in range(self.maxiter):
                start_time = time()
                # grad_f = self._ComputePenaltyGrad(beta[i], D, self.mu)
                mu = np.exp(np.clip(np.dot(X, beta[i]), -np.infty, M))
                grad = -X.T.dot(y - mu) / n + np.dot(D, beta[i])

                w = np.diag(mu)
                hinv = np.linalg.inv(X.T.dot(w).dot(X) / n + D)

                beta.append(beta[i] - np.dot(hinv, grad))

                time_traj.append(time() - start_time)
                if np.linalg.norm(grad) < self.tol:
                    break

        elif self.family == 'multinomial':

            m = y.shape[1]
            y_ = y[:, :-1]

            beta = [np.zeros((p, m - 1))]
            for i in range(self.maxiter):

                Xb = np.clip(np.dot(X, beta[i]), -M, M)
                mu = np.array([np.exp(xb) / (1 + np.exp(xb).sum()) for xb in Xb])
                # mu = np.column_stack([mu, 1.-mu.sum(1)])
                grad = -X.T.dot(y_ - mu) / n + np.dot(D, beta[i])

                if np.linalg.norm(grad) < self.tol:
                    break

                mu = np.clip(mu, 0 + 1e-10, 1 - 1e-10)
                beta.append(np.zeros((p, m - 1)))
                for j in range(m - 1):
                    w = np.diag(mu[:, j] * (1 - mu[:, j]))
                    h = X.T.dot(w).dot(X) / n + D + 1e-3 * np.identity(p)
                    hinv = np.linalg.inv(h)
                    beta[i + 1][:, j] = beta[i][:, j] - np.dot(hinv, grad[:, j])

        self.time_traj = np.cumsum(time_traj)
        self.N_time = self.time_traj[-1]
        self.coef_traj = beta
        self.coef_ = beta[-1]

    def _predict(self, X):
        X, A = self._process_predictors(X)
        if self.fit_intercept:
            X = np.column_stack([np.ones(len(X)), X])

        if self.family == 'normal':
            return np.dot(X, self.coef_)
        elif self.family == 'bernoulli':
            Xb = np.dot(X, self.coef_)
            p = 1 / (1 + np.exp(-Xb))
            return (p > 0.5) * 1.
        elif self.family == 'poisson':
            Xb = np.dot(X, self.coef_)
            return np.exp(Xb)
        elif self.family == 'multinomial':
            Xb = np.dot(X, self.coef_)
            mu = np.array([np.exp(xb) / (1 + np.exp(xb).sum()) for xb in Xb])
            mu = np.column_stack([mu, 1. - mu.sum(1)])
            return self._enc.inverse_transform(mu).flatten()

    def _predict_proba(self, X):
        if self.fit_intercept:
            X = np.column_stack([np.ones(len(X)), X])

        if self.family == 'normal':
            return np.dot(X, self.coef_)
        elif self.family == 'bernoulli':
            Xb = np.dot(X, self.coef_)
            p = 1 / (1 + np.exp(-Xb))
            return np.column_stack([1 - p, p])
        elif self.family == 'poisson':
            Xb = np.dot(X, self.coef_)
            return np.exp(Xb)
        elif self.family == 'multinomial':
            Xb = np.dot(X, self.coef_)
            mu = np.array([np.exp(xb) / (1 + np.exp(xb).sum()) for xb in Xb])
            return np.column_stack([mu, 1. - mu.sum(1)])

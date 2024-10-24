
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

from scipy.special import expit as sigmoid
from scipy.special import gammaln


def clipped_sigmoid(z, eps=1e-6):
    return np.clip(sigmoid(z), a_min=eps, a_max=1.-eps)

def clipped_exp(z, eps=10):
    return np.clip(np.exp(z), a_min=None, a_max=eps)

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
            maxiter=1000,
            tol=1e-6
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

        ls_grid = np.exp(np.linspace(np.log(1e-3), np.log(1), 20))
        beta = np.zeros(p)
        time_traj = [0.]

        conj_fn = lambda b_n, b_o: np.max([0, b_n @ (b_n - b_o) / np.clip(b_o @ b_o, a_min=1e-4, a_max=None)])
        if self.family == 'normal':
            loss_fn = lambda b: .5*np.square(y - X @ b).mean() + .5*b @ D @ b
            grad_fn = lambda b: -X.T @ (y - X @ b) / n + D @ b

            grad_old = grad_fn(beta)
            conj_old = np.copy(grad_old)
            for i in range(self.maxiter):
                start_time = time()
                grad = grad_fn(beta)
                if np.linalg.norm(grad) < self.tol:
                    break

                conj = grad + conj_fn(grad, grad_old)*conj_old
                cand = [beta - conj * v for v in ls_grid]
                ls_min = np.argmin([loss_fn(c) for c in cand])
                beta = cand[ls_min]

                time_traj.append(time() - start_time)
                grad_old = np.copy(grad)
                conj_old = np.copy(conj)

        elif self.family == 'bernoulli':
            self.classes_ = unique_labels(y)
            def loss_fn(b):
                xb = X@b
                return -np.sum(y*xb - np.log(1 + np.exp(xb))) / n + .5*b @ D @ b
            grad_fn = lambda b: -X.T @ (y - clipped_sigmoid(X@b)) / n + D @ b

            grad_old = grad_fn(beta)
            conj_old = np.copy(grad_old)
            for i in range(self.maxiter):
                start_time = time()
                grad = grad_fn(beta)
                if np.linalg.norm(grad) < self.tol:
                    break

                conj = grad + conj_fn(grad, grad_old)*conj_old
                cand = [beta - conj * v for v in ls_grid]
                ls_min = np.argmin([loss_fn(c) for c in cand])
                beta = cand[ls_min]

                time_traj.append(time() - start_time)

                grad_old = np.copy(grad)
                conj_old = np.copy(conj)

        elif self.family == 'poisson':
            def loss_fn(b):
                xb = X@b
                return -np.sum(y*xb - clipped_exp(xb) - gammaln(y+1)) / n + .5*b @ D @ b
            grad_fn = lambda b: -X.T @ (y - clipped_exp(X@b)) / n + D @ b

            grad_old = grad_fn(beta)
            conj_old = np.copy(grad_old)
            for i in range(self.maxiter):
                start_time = time()
                grad = grad_fn(beta)
                if np.linalg.norm(grad) < self.tol:
                    break
                conj = grad + conj_fn(grad, grad_old)*conj_old
                cand = [beta - conj * v for v in ls_grid]
                ls_min = np.argmin([loss_fn(c) for c in cand])
                beta = cand[ls_min]

                time_traj.append(time() - start_time)

                grad_old = np.copy(grad)
                conj_old = np.copy(conj)

        elif self.family == 'multinomial':
            m = y.shape[1]
            y_ = y[:, :-1]
            def loss_fn(b):
                mu = np.array([np.exp(xb) / (1 + np.exp(xb).sum()) for xb in X @ b])
                mu = np.clip(mu, 0 + 1e-6, 1 - 1e-6)
                return -np.sum(y_ * np.log(mu)) / n + .5 * np.sum([b_ @ D @ b_ for b_ in b.T])
            def grad_fn(b):
                mu = np.array([np.exp(xb) / (1 + np.exp(xb).sum()) for xb in X @ b])
                mu = np.clip(mu, 0 + 1e-6, 1 - 1e-6)
                return -X.T.dot(y_ - mu) / n + D @ b

            beta = np.zeros((p, m-1))
            grad_old = grad_fn(beta)
            conj_old = np.copy(grad_old)
            for i in range(self.maxiter):
                start_time = time()
                grad = grad_fn(beta)
                if np.linalg.norm(grad) < self.tol:
                    break
                conj = (grad.flatten() + conj_fn(grad.flatten(), grad_old.flatten())*conj_old.flatten()).reshape(p, m-1)
                cand = [beta - conj * v for v in ls_grid]
                ls_min = np.argmin([loss_fn(c) for c in cand])
                beta = cand[ls_min]

                time_traj.append(time() - start_time)

                grad_old = np.copy(grad)
                conj_old = np.copy(conj)


        self.time_traj = np.cumsum(time_traj)
        self.N_time = self.time_traj[-1]
        self.coef_ = beta

    def _predict(self, X):
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

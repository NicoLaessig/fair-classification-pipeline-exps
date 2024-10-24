
'''
    Implementation of the General Fair Empirical Risk Minimization method proposed in
    Oneto, L., Donini, M., & Pontil, M. (2020, July). General fair empirical risk minimization. In 2020 International Joint Conference on Neural Networks (IJCNN) (pp. 1-8). IEEE.
'''

import numpy as np
import cvxpy as cp
from itertools import combinations
from . import BaseFairEstimator
from ..util import discretization

class GeneralFairERM(BaseFairEstimator):
    def __init__(
            self,
            sensitive_index=0,
            sensitive_predictor=False,
            standardize=False,
            eps=0.,
            K=10,
            fit_intercept=True):
        assert eps >= 0, 'parameter eps must be nonnegative real value'

        super().__init__(
            sensitive_index=sensitive_index,
            sensitive_predictor=sensitive_predictor,
            standardize=standardize
        )

        self.eps = eps
        self.K = K
        self.fit_intercept = fit_intercept
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
        np.random.seed(1234)
        beta = cp.Variable(p)
        
        objective = cp.sum(cp.square(y - X@beta)) / n
        
        # add L2 norm regularizer (suspended now)
        # objective += self.lam*cp.norm(beta, 2)

        YD = discretization(y, A, 'normal', self.K)

        D = []
        for yd in set(YD):
            for a, b in combinations(set(A), 2):
                Xay = X[np.logical_and(A == a, YD == yd)]
                Xby = X[np.logical_and(A == b, YD == yd)]
                if len(Xay)*len(Xby) > 0:
                    D.append([Xay.mean(0), -Xby.mean(0)])

        D = np.row_stack(D)
        
        constraints = [cp.norm(D@beta, 1) <= self.eps]
        problem = cp.Problem(cp.Minimize(objective), constraints)

        #solver = cp.GUROBI if 'GUROBI' in cp.installed_solvers() else cp.ECOS
        solver = cp.ECOS
        problem.solve(solver=solver)
        self.coef_ = np.array(beta.value).flatten()
        
    def _predict(self, X):
        X, A = self._process_predictors(X)
        if self.fit_intercept:
            X = np.column_stack([np.ones(len(X)), X])
        return X.dot(self.coef_).flatten()

    def _predict_proba(self, X):
        raise NotImplementedError

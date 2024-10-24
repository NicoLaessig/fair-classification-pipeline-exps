
'''
    Implementations of the Fair Constraints and Disparate Mistreatment Methods in
    Zafar, M. B., Valera, I., Rogriguez, M. G., & Gummadi, K. P. (2017, April). Fairness constraints: Mechanisms for fair classification. In Artificial Intelligence and Statistics (pp. 962-970). PMLR.
    and
    Zafar, M. B., Valera, I., Gomez Rodriguez, M., & Gummadi, K. P. (2017, April). Fairness beyond disparate treatment & disparate impact: Learning classification without disparate mistreatment. In Proceedings of the 26th international conference on world wide web (pp. 1171-1180).

    Original implementations of these methods can be found on https://github.com/mbilalzafar/fair-classification
'''


import sys
import traceback
import dccp
import numpy as np
import cvxpy as cp
from sklearn.preprocessing import OneHotEncoder
from . import BaseFairEstimator


class FairnessConstraintModel(BaseFairEstimator):
    def __init__(
            self,
            sensitive_index=0,
            sensitive_predictor=False,
            standardize=False,
            c=0.,
            #lam=0.,
            fit_intercept=True,
            tau=0.5, mu=1.2, eps=1e-4,
            max_iters=100, max_iter_dccp=50):

        #assert lam >= 0, 'parameter lam must be positive real value'
        assert c >= 0, 'paramter c must be nonnegative real value'

        super().__init__(
            sensitive_index=sensitive_index,
            sensitive_predictor=sensitive_predictor,
            standardize=standardize
        )

        #self.lam = lam
        self.c = c
        self.fit_intercept = fit_intercept
        self.tau = tau
        self.mu = mu
        self.eps = eps
        self.max_iters = max_iters
        self.max_iter_dccp = max_iter_dccp
        self.coef_ = None
        self.loss = 'logistic'

    def fit(self, X, y):
        self.status = True

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
        beta.value = np.zeros(p)#np.random.uniform(-1, 1, p)

        objective = 0.

        if self.loss == 'logistic':
            # logistic regression negative log-likelihood
            objective += -cp.sum(cp.multiply(y, X @ beta) - cp.logistic(X @ beta)) / n
        else:
            # linear SVM loss
            objective += cp.sum(cp.maximum(0, 1 - cp.multiply(y, X @ beta))) / n

        # add L2 norm regularizer (suspended now)
        # objective += self.lam*cp.norm(beta, 2)

        constraints = self.get_di_cons_single_boundary(X, y, A, beta, self.c)
        problem = cp.Problem(cp.Minimize(objective), constraints)

        try:

            problem.solve(method='dccp', verbose=False, tau=self.tau, mu=self.mu, tau_max=1e10,
                          feastol=self.eps, reltol=self.eps, feastol_inacc=self.eps, reltol_inacc=self.eps,
                          max_iters=self.max_iters, max_iter=self.max_iter_dccp, solver=cp.ECOS)

            #print("Optimization done, problem status:", problem.status)
            #assert problem.status == "Converged" or problem.status == "optimal"

            # check that the fairness constraint is satisfied
            '''
            for f_c in constraints:
                try:
                    assert f_c.value == True
                except:
                    print("Assertion failed. Fairness constraints not satisfied.")
                    print(traceback.print_exc())
                    sys.stdout.flush()
                    return
                    # sys.exit(1)
            '''
            self.coef_ = np.array(beta.value).flatten()

        except:
            print('ECOS failed')
            self.coef_ = None
            self.status = None


    def _predict(self, X):
        X, A = self._process_predictors(X)
        if self.fit_intercept:
            X = np.column_stack([np.ones(len(X)), X])
        p = 1 / (1 + np.exp(-np.dot(X, self.coef_).flatten()))
        return (p > 0.5) * 1

    def _predict_proba(self, X):
        if self.fit_intercept:
            X = np.column_stack([np.ones(len(X)), X])
        p = 1 / (1 + np.exp(-np.dot(X, self.coef_).flatten()))
        return np.column_stack([1 - p, p])

    def get_di_cons_single_boundary(self, X, y, A, beta, c):

        """
        Parity impact constraint
        """

        assert c >= 0  # covariance thresh has to be a small positive number

        n, p = X.shape
        constraints = []
        z_i_z_bar = A - np.mean(A)

        fx = X @ beta
        prod = cp.sum(cp.multiply(z_i_z_bar, fx)) / n

        constraints.append(prod <= c)
        constraints.append(prod >= -c)

        return constraints


class DisparateMistreatmentModel(BaseFairEstimator):
    def __init__(
            self,
            sensitive_index=0,
            sensitive_predictor=False,
            standardize=False,
            c=0.,
            #lam=0.,
            fit_intercept=True,
            tau=0.5, mu=1.2, eps=1e-4,
            max_iters=100, max_iter_dccp=50):

        #assert lam >= 0, 'parameter lam must be positive real value'
        assert c >= 0, 'paramter c must be nonnegative real value'

        super().__init__(
            sensitive_index=sensitive_index,
            sensitive_predictor=sensitive_predictor,
            standardize=standardize
        )

        #self.lam = lam
        self.c = c
        self.fit_intercept = fit_intercept
        self.tau = tau
        self.mu = mu
        self.eps = eps
        self.max_iters = max_iters
        self.max_iter_dccp = max_iter_dccp
        self.coef_ = None
        self.loss = 'logistic'


    def fit(self, X, y):
        self.status = True

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
        beta.value = np.zeros(p) #np.random.uniform(-1, 1, p)

        objective = 0.

        if self.loss == 'logistic':
            # logistic regression negative log-likelihood
            objective += -cp.sum(cp.multiply(y, X @ beta) - cp.logistic(X @ beta)) / n
        else:
            # linear SVM loss
            objective += cp.sum(cp.maximum(0, 1 - cp.multiply(y, X @ beta))) / n

        # add L2 norm regularizer
        # objective += self.lam*cp.norm(beta, 2)

        constraints = self.get_di_cons_single_boundary(X, y, A, beta, self.c)
        problem = cp.Problem(cp.Minimize(objective), constraints)

        try:

            problem.solve(method='dccp', verbose=False, tau=self.tau, mu=self.mu, tau_max=1e10,
                          feastol=self.eps, reltol=self.eps, feastol_inacc=self.eps, reltol_inacc=self.eps,
                          max_iters=self.max_iters, max_iter=self.max_iter_dccp, solver=cp.ECOS)

            #print("Optimization done, problem status:", problem.status)
            #assert problem.status == "Converged" or problem.status == "optimal"

            # check that the fairness constraint is satisfied
            '''
            for f_c in constraints:
                try:
                    assert f_c.value == True
                except:
                    print("Assertion failed. Fairness constraints not satisfied.")
                    print(traceback.print_exc())
                    sys.stdout.flush()
                    return
                    # sys.exit(1)
            '''
            self.coef_ = np.array(beta.value).flatten()

        except:
            print('ECOS failed')
            self.coef_ = None
            self.status = None


    def _predict(self, X):
        X, A = self._process_predictors(X)
        if self.fit_intercept:
            X = np.column_stack([np.ones(len(X)), X])
        p = 1 / (1 + np.exp(-np.dot(X, self.coef_).flatten()))
        return (p > 0.5) * 1

    def _predict_proba(self, X):
        if self.fit_intercept:
            X = np.column_stack([np.ones(len(X)), X])
        p = 1 / (1 + np.exp(-np.dot(X, self.coef_).flatten()))
        return np.column_stack([1 - p, p])

    def get_di_cons_single_boundary(self, X, y, A, beta, c):

        """
        Parity impact constraint
        """

        assert c >= 0  # covariance thresh has to be a small positive number

        n, p = X.shape
        constraints = []

        A_enc = OneHotEncoder(sparse=False, drop='if_binary').fit_transform(A.reshape(-1, 1))

        for A_ in A_enc.T:
            s_val_to_total = {ct: {} for ct in [0, 1, 2]}
            s_val_to_avg = {ct: {} for ct in [0, 1, 2]}
            cons_sum_dict = {ct: {} for ct in [0, 1, 2]}

            for a in set(A_):
                s_val_to_total[0][a] = sum(A_ == a)
                s_val_to_total[1][a] = sum(np.logical_and(A_ == a, y == 0))  # FPR constraint so we only consider the ground truth negative dataset for computing the covariance
                s_val_to_total[2][a] = sum(np.logical_and(A_ == a, y == 1))

            for ct in [0, 1, 2]:
                s_val_to_avg[ct][0] = s_val_to_total[ct][1] / float(s_val_to_total[ct][0] + s_val_to_total[ct][
                    1])  # N1/N in our formulation, differs from one constraint type to another
                s_val_to_avg[ct][1] = 1.0 - s_val_to_avg[ct][0]  # N0/N

            for a in set(A_):
                idx = (A_ == a)

                dist_bound_prod = cp.multiply((y[idx] * 2 - 1), X[idx] @ beta)

                cons_sum_dict[0][a] = cp.sum(cp.minimum(0, dist_bound_prod)) * (
                            s_val_to_avg[0][a] / len(X))  # avg misclassification distance from boundary
                cons_sum_dict[1][a] = cp.sum(cp.minimum(0, cp.multiply((1 - y[idx]), dist_bound_prod))) * (
                            s_val_to_avg[1][a] / sum(y == 0))  # avg false positive distance from boundary (only operates on the ground truth neg dataset)
                cons_sum_dict[2][a] = cp.sum(cp.minimum(0, cp.multiply((0 + y[idx]), dist_bound_prod))) * (
                            s_val_to_avg[2][a] / sum(y == 1))  # avg false negative distance from boundary

            cts = [1, 2]  # FPR and FNR
            for ct in cts:
                constraints.append(cons_sum_dict[ct][1] <= cons_sum_dict[ct][0] + c)
                constraints.append(cons_sum_dict[ct][1] >= cons_sum_dict[ct][0] - c)

        return constraints



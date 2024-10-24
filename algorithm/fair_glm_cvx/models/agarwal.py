
'''
    Implementation of the Reductions Approach proposed in
    Agarwal, A., Beygelzimer, A., Dudík, M., Langford, J., & Wallach, H. (2018, July). A reductions approach to fair classification. In International Conference on Machine Learning (pp. 60-69). PMLR.
    and
    Agarwal, A., Dudík, M., & Wu, Z. S. (2019, May). Fair regression: Quantitative definitions and reduction-based algorithms. In International Conference on Machine Learning (pp. 120-129). PMLR.
    based on the fairlearn implementation https://github.com/fairlearn/fairlearn
'''

import numpy as np
from . import BaseFairEstimator
from fairlearn.reductions import GridSearch
#from fairlearn.reductions import DemographicParity, BoundedGroupLoss, ZeroOneLoss, SquareLoss
from sklearn.linear_model import LogisticRegression, LinearRegression


class ReductionsApproach(BaseFairEstimator):
    def __init__(
            self,
            sensitive_index=0,
            sensitive_predictor=False,
            standardize=False,
            penalty='SP',
            family='bernoulli',
            c=0.,
            fit_intercept=True
            ):

        super().__init__(
            sensitive_index=sensitive_index,
            sensitive_predictor=sensitive_predictor,
            standardize=standardize
        )

        assert family in ['bernoulli', 'normal']
        assert penalty in ['SP', 'BGL']

        self.c = c
        self.fit_intercept = fit_intercept
        self.family = family
        self.penalty = penalty
        self._model = None

    def fit(self, X, y):
        y = y.astype(np.float64)

        assert type(y) == np.ndarray
        assert type(X) == np.ndarray
        assert len(X) == len(y)

        X, A = self._process_predictors(X)

        if self.fit_intercept:
            X = np.column_stack([np.ones(len(X)), X])

        if self.family == 'bernoulli':
            basemodel = LogisticRegression(
                penalty='none',
                fit_intercept=False
            )
        else:
            basemodel = LinearRegression(
                fit_intercept=False
            )

        if self.penalty == 'SP':
            constraint = DemographicParity(difference_bound=0)
        else:
            if self.family == 'normal':
                constraint = BoundedGroupLoss(SquareLoss(y.min(), y.max()), upper_bound=0)
            else:
                constraint = BoundedGroupLoss(ZeroOneLoss(), upper_bound=0)

        model = GridSearch(
            basemodel, constraint_weight=self.c,
            grid_size=36,
            constraints=constraint
        )

        model.fit(X, y, sensitive_features=A)
        self._model = model

    def _predict(self, X):
        if self.fit_intercept:
            X = np.column_stack([np.ones(len(X)), X])
        return self._model.predict(X)

    def _predict_proba(self, X):
        if self.fit_intercept:
            X = np.column_stack([np.ones(len(X)), X])
        return self._model.predict_proba(X)

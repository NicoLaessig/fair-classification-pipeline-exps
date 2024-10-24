import pandas as pd
from aif360.sklearn.inprocessing import GridSearchReduction
from sklearn.base import BaseEstimator


class ReductionClassifier(BaseEstimator):
    def __init__(self, estimator, constraints, random_state=None, **kwargs):
        self.estimator = estimator
        self.random_state = random_state
        self.constraints = constraints
        self.clf = GridSearchReduction(
            prot_attr=0, estimator=estimator, constraints=constraints, **kwargs
        )
        self.set_params(random_state=random_state)

    def set_params(self, **kwargs):
        try:
            self.clf.estimator.set_params(**kwargs)
        except:
            pass

    def fit(
        self,
        X_train,
        y_train,
    ):
        self.clf.fit(pd.DataFrame(X_train), y_train)
        return self

    def predict(self, X_test):
        return self.clf.predict(pd.DataFrame(X_test))

    def predict_proba(self, X_test):
        return self.clf.predict_proba(pd.DataFrame(X_test))

from sklearn.base import BaseEstimator
from fairlearn.postprocessing import ThresholdOptimizer


class ThresholdClassifier(BaseEstimator):
    def __init__(
        self, estimator, constraints, random_state=None, **fairens_kwargs
    ) -> None:
        self.estimator = estimator
        self.estimator_ = estimator
        self.constraints = constraints
        self.fairens_kwargs = fairens_kwargs
        self.set_params(random_state=random_state)

    def set_params(self, **kwargs):
        try:
            self.random_state = kwargs['random_state']
            self.estimator_.set_params(**kwargs)
        except:
            pass

    def fit(self, X, y, sensitive_features):
        self.estimator_.fit(X, y, sensitive_features)
        self.postprocessor = ThresholdOptimizer(
            estimator=self.estimator_,
            constraints=self.constraints,
            prefit=True,
        )
        self.postprocessor.fit(X, y, sensitive_features=sensitive_features)

    def predict(self, X, sensitive_features, random_state=None):
        random_state = self.random_state if random_state is None else random_state
        return self.postprocessor.predict(
            X, sensitive_features=sensitive_features, random_state=random_state
        )

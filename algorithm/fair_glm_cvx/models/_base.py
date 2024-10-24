
import numpy as np

from sklearn.base import BaseEstimator
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler


class BaseFairEstimator(BaseEstimator):
    def __init__(self, sensitive_index=0, sensitive_predictor=False, standardize=False, **kwargs):
        self.sensitive_index = sensitive_index
        self.sensitive_predictor = sensitive_predictor
        self.sensitive_attributes = None
        self.num_sensitive_attributes = None
        self.standardize = standardize

        self._sensitive_encoder = OneHotEncoder(sparse=False, drop='first') if sensitive_predictor else None
        self._feature_scaler = StandardScaler() if standardize else None
        self.status = True

    def fit(self, X, y):
        raise NotImplementedError

    def _predict(self, X):
        raise NotImplementedError

    def _predict_proba(self, X):
        raise NotImplementedError

    def predict(self, X):
        A = None
        if self.sensitive_predictor:
            A = self._sensitive_encoder.transform(X[:, self.sensitive_index])
            X = np.delete(X, self.sensitive_index, axis=1)

        if self.standardize:
            X = self._feature_scaler.transform(X)

        X = np.column_stack([A, X]) if A else X
        return self._predict(X)


    def predict_proba(self, X):
        A = None
        if self.sensitive_predictor:
            A = self._sensitive_encoder.transform(X[:, self.sensitive_index])
            X = np.delete(X, self.sensitive_index, axis=1)

        if self.standardize:
            X = self._feature_scaler.transform(X)

        X = np.column_stack([A, X]) if A else X
        return self._predict_proba(X)

    def _process_predictors(self, X):
        A = X[:, self.sensitive_index]
        X = np.delete(X, self.sensitive_index, axis=1)

        if self.standardize:
            X = self._feature_scaler.fit_transform(X)
            
        if self.sensitive_predictor:
            X = np.column_stack([
                self._sensitive_encoder.fit_transform(A), X])

        self.sensitive_attributes = list(set(A))
        self.num_sensitive_attributes = len(self.sensitive_attributes)

        return X, A


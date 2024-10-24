"""
In this python file multiple classification models are trained.
"""
import pandas as pd
import numpy as np
import copy
from aif360.algorithms.postprocessing import *
from aif360.datasets import BinaryLabelDataset
from sklearn.model_selection import train_test_split
from .evaluation.eval_classifier import acc_bias


class AIFPostprocessing():
    """Multiple different model learners are part of this class.

    Parameters
    ----------
    X_train: {array-like, sparse matrix}, shape (n_samples, m_features)
        Training data vector, where n_samples is the number of samples and
        m_features is the number of features.

    y_train: array-like, shape (n_samples)
        Label vector relative to the training data X_train.

    X_test: {array-like, sparse matrix}, shape (n_samples, m_features)
        Test data vector, where n_samples is the number of samples and
        m_features is the number of features.

    y_test: array-like, shape (n_samples)
        Label vector relative to the test data X_test.

    sens_attrs: list of strings
        List of the column names of sensitive attributes in the dataset.

    ...
    """
    def __init__(self, X_train, X_test, y_train, y_test, sens_attrs,
        dataset_train, dataset_test, privileged_groups, unprivileged_groups,
        label, metric, link, remove):
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test
        self.sens_attrs = sens_attrs
        self.dataset_train = dataset_train
        self.dataset_test = dataset_test
        self.privileged_groups = privileged_groups
        self.unprivileged_groups = unprivileged_groups
        self.label = label
        self.metric = metric
        self.link = link
        self.remove = remove


    def eqodds_postproc(self, classifier, do_eval=False):
        """
        ...
        """
        dataset_orig_train, dataset_orig_valid = self.dataset_train.split([0.4], shuffle=True)
        X_train = dataset_orig_train.features
        y_train = dataset_orig_train.labels.ravel()
        X_valid = dataset_orig_valid.features
        y_valid = dataset_orig_valid.labels.ravel()
        X_test = self.dataset_test.features
        y_test = self.dataset_test.labels.ravel()
        if self.remove:
            X_train = dataset_orig_train.convert_to_dataframe()[0]
            X_train = X_train.loc[:, X_train.columns != self.label]
            X_valid = dataset_orig_valid.convert_to_dataframe()[0]
            X_valid = X_valid.loc[:, X_valid.columns != self.label]
            X_test = copy.deepcopy(self.X_test)
            for sens in self.sens_attrs:
                X_train = X_train.drop(sens, axis=1)
                X_valid = X_valid.drop(sens, axis=1)
                X_test = X_test.drop(sens, axis=1)
        dataset_orig_valid_pred = copy.deepcopy(dataset_orig_valid)
        classifier.fit(X_train, y_train)
        prediction = classifier.predict_proba(X_valid)[:,1]
        dataset_orig_valid_pred.scores = prediction.reshape(-1, 1)
        dataset_orig_valid_pred.labels = classifier.predict(X_valid).reshape(-1, 1)

        prediction = classifier.predict_proba(X_test)[:,1]
        dataset_orig_test_pred = copy.deepcopy(self.dataset_test)
        dataset_orig_test_pred.scores = prediction.reshape(-1, 1)
        dataset_orig_test_pred.labels = classifier.predict(X_test).reshape(-1, 1)

        model = EqOddsPostprocessing(self.unprivileged_groups, self.privileged_groups)
        model.fit(dataset_orig_valid, dataset_orig_valid_pred)
        pred = list(model.predict(dataset_orig_test_pred).labels.ravel())

        if do_eval:
            metric_val = acc_bias(self.dataset_test, np.asarray(pred).reshape(-1,1), self.unprivileged_groups, self.privileged_groups, self.metric)
        else:
            metric_val = None


        return pred, metric_val, "EqOddsPostprocessing", model


    def calibrated_eqodds_postproc(self, classifier, do_eval=False):
        """
        ...
        """
        dataset_orig_train, dataset_orig_valid = self.dataset_train.split([0.3], shuffle=True)
        X_train = dataset_orig_train.features
        y_train = dataset_orig_train.labels.ravel()
        X_valid = dataset_orig_valid.features
        y_valid = dataset_orig_valid.labels.ravel()
        X_test = self.dataset_test.features
        y_test = self.dataset_test.labels.ravel()

        if self.remove:
            X_train = dataset_orig_train.convert_to_dataframe()[0]
            X_train = X_train.loc[:, X_train.columns != self.label]
            X_valid = dataset_orig_valid.convert_to_dataframe()[0]
            X_valid = X_valid.loc[:, X_valid.columns != self.label]
            X_test = copy.deepcopy(self.X_test)
            for sens in self.sens_attrs:
                X_train = X_train.drop(sens, axis=1)
                X_valid = X_valid.drop(sens, axis=1)
                X_test = X_test.drop(sens, axis=1)
        dataset_orig_valid_pred = copy.deepcopy(dataset_orig_valid)
        classifier.fit(X_train, y_train)
        prediction = classifier.predict_proba(X_valid)[:,1]
        dataset_orig_valid_pred.scores = prediction.reshape(-1, 1)
        
        prediction = classifier.predict_proba(X_test)[:,1]
        dataset_orig_test_pred = copy.deepcopy(self.dataset_test)
        dataset_orig_test_pred.scores = prediction.reshape(-1, 1)

        model = CalibratedEqOddsPostprocessing(self.unprivileged_groups, self.privileged_groups, cost_constraint='weighted')
        model.fit(dataset_orig_valid, dataset_orig_valid_pred)
        pred = list(model.predict(dataset_orig_test_pred).labels.ravel())

        if do_eval:
            metric_val = acc_bias(self.dataset_test, np.asarray(pred).reshape(-1,1), self.unprivileged_groups, self.privileged_groups, self.metric)
        else:
            metric_val = None

        return pred, metric_val, "CalibratedEqOddsPostprocessing", model


    def reject_option_class(self, classifier, eps, do_eval=False):
        """
        ...
        """
        if self.metric == "demographic_parity":
            metric = "Statistical parity difference"
        elif self.metric == "equalized_odds":
            metric = "Average odds difference"
        elif self.metric == "equal_opportunity":
            metric = "Equal opportunity difference"
        else:
            metric = "Statistical parity difference"

        dataset_orig_train, dataset_orig_valid = self.dataset_train.split([0.3], shuffle=True)
        X_train = dataset_orig_train.features
        y_train = dataset_orig_train.labels.ravel()
        X_valid = dataset_orig_valid.features
        y_valid = dataset_orig_valid.labels.ravel()
        dataset_orig_valid_pred = copy.deepcopy(dataset_orig_valid)
        X_test = self.dataset_test.features
        y_test = self.dataset_test.labels.ravel()
        if self.remove:
            X_train = dataset_orig_train.convert_to_dataframe()[0]
            X_train = X_train.loc[:, X_train.columns != self.label]
            X_valid = dataset_orig_valid.convert_to_dataframe()[0]
            X_valid = X_valid.loc[:, X_valid.columns != self.label]
            X_test = copy.deepcopy(self.X_test)
            for sens in self.sens_attrs:
                X_train = X_train.drop(sens, axis=1)
                X_valid = X_valid.drop(sens, axis=1)
                X_test = X_test.drop(sens, axis=1)
                
        classifier.fit(X_train, y_train)
        prediction = classifier.predict_proba(X_valid)[:,1]
        dataset_orig_valid_pred.scores = prediction.reshape(-1, 1)

        
        prediction = classifier.predict_proba(X_test)[:,1]
        dataset_orig_test_pred = copy.deepcopy(self.dataset_test)
        dataset_orig_test_pred.scores = prediction.reshape(-1, 1)

        model = RejectOptionClassification(self.unprivileged_groups, self.privileged_groups,
            metric_name=metric, metric_ub=eps, metric_lb=-eps)
        dataset_cleaned = model.fit(dataset_orig_valid, dataset_orig_valid_pred)
        pred = list(model.predict(dataset_orig_test_pred).labels.ravel())

        if do_eval:
            metric_val = acc_bias(self.dataset_test, np.asarray(pred).reshape(-1,1), self.unprivileged_groups, self.privileged_groups, self.metric)
        else:
            metric_val = None

        return pred, metric_val, "RejectOptionClassification", model

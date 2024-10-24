"""
In this python file multiple classification models are trained.
"""
import numpy as np
import pandas as pd
import copy
from aif360.algorithms.preprocessing import *
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from aif360.datasets import BinaryLabelDataset
from aif360.metrics import BinaryLabelDatasetMetric
from .evaluation.eval_classifier import acc_bias


class AIFPreprocessing():
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
    """
    def __init__(self, dataset, dataset_train, dataset_test, X_test, testsize, randomstate,
        sens_attrs, privileged_groups, unprivileged_groups, label, metric, link, remove):
        self.dataset = dataset
        self.dataset_train = dataset_train
        self.dataset_test = dataset_test
        self.X_test = X_test
        self.testsize = testsize
        self.randomstate = randomstate
        self.sens_attrs = sens_attrs
        self.privileged_groups = privileged_groups
        self.unprivileged_groups = unprivileged_groups
        self.label = label
        self.metric = metric
        self.link = link
        self.remove = remove


    def disparate_impact_remover(self, classifier, repair, do_eval=False):
        """
        ...
        """
        dataset_train = copy.deepcopy(self.dataset_train)
        dataset_test = copy.deepcopy(self.dataset_test)
        index = self.dataset_train.feature_names.index(self.sens_attrs[0])

        model = DisparateImpactRemover(repair_level=repair)

        train_repd = model.fit_transform(dataset_train)
        test_repd = model.fit_transform(dataset_test)
        
        X_train = np.delete(train_repd.features, index, axis=1)
        X_test = np.delete(test_repd.features, index, axis=1)
        y_train = train_repd.labels.ravel()

        classifier.fit(X_train, y_train)
        test_repd_pred = test_repd.copy()
        pred = classifier.predict(X_test)

        if do_eval:
            metric_val = acc_bias(self.dataset_test, np.asarray(pred).reshape(-1,1), self.unprivileged_groups, self.privileged_groups, self.metric)
        else:
            metric_val = None

        cols = train_repd.feature_names.remove(self.sens_attrs[0])
        X_train = pd.DataFrame(X_train, columns=train_repd.feature_names)
        y_train = pd.DataFrame(np.asarray(y_train).reshape(-1,1), columns=[self.label])

        return pred, X_train, y_train, X_test, metric_val, "DisparateImpactRemover", classifier


    def lfr(self, classifier, k, Ay, Az, do_eval=False):
        """
        ...
        """
        scale_orig = StandardScaler()
        dataset = copy.deepcopy(self.dataset)
        dataset.features = scale_orig.fit_transform(dataset.features)

        model = LFR(self.unprivileged_groups, self.privileged_groups, k=10, Ay=Ay, Az=Az)
        model = model.fit(self.dataset_train)
        dataset_transf_train = model.transform(self.dataset_train)
        dataset_transf_test = model.transform(self.dataset_test)

        X_train = dataset_transf_train.convert_to_dataframe()[0]
        y_train = X_train[self.label]
        X_train = X_train.loc[:, X_train.columns != self.label]

        X_test = dataset_transf_test.convert_to_dataframe()[0]
        X_test = X_test.loc[:, X_test.columns != self.label]
        
        if self.remove:  
            for sens in self.sens_attrs:
                X_train = X_train.drop(sens, axis=1)
                X_test = X_test.drop(sens, axis=1)
            try:
                classifier.fit(X_train, y_train)
                pred = classifier.predict(X_test)
            except:
                #y_train has only one value
                preds = dataset_transf_test.labels
                pred = [preds[i][0] for i in range(len(preds))]
        else:
            preds = dataset_transf_test.labels
            pred = [preds[i][0] for i in range(len(preds))]

        if do_eval:
            metric_val = acc_bias(self.dataset_test, preds, self.unprivileged_groups, self.privileged_groups, self.metric)
        else:
            metric_val = None

        return pred, X_train, y_train, X_test, metric_val, "LFR", classifier


    def reweighing(self, classifier, do_eval=False):
        """
        ...
        """
        model = Reweighing(self.unprivileged_groups, self.privileged_groups)
        dataset_cleaned = model.fit_transform(self.dataset_train)

        dataset_cleaned, attr = dataset_cleaned.convert_to_dataframe()
        X_train = dataset_cleaned.loc[:, dataset_cleaned.columns != self.label]
        y_train = dataset_cleaned[self.label]


        X_test = copy.deepcopy(self.X_test)

        if self.remove:
            for sens in self.sens_attrs:
                X_train = X_train.drop(sens, axis=1)
                X_test = X_test.drop(sens, axis=1)

        classifier.fit(X_train, y_train, attr["instance_weights"])
        pred = classifier.predict(X_test)

        if do_eval:
            metric_val = acc_bias(self.dataset_test, np.asarray(pred).reshape(-1,1), self.unprivileged_groups, self.privileged_groups, self.metric)
        else:
            metric_val = None

        return pred, X_train, y_train, attr["instance_weights"], metric_val, "Reweighing", classifier

"""
In this python file multiple classification models are trained.
"""
import numpy as np
import tensorflow as tf
from aif360.datasets import BinaryLabelDataset
from aif360.algorithms.inprocessing import *
from .evaluation.eval_classifier import acc_bias


class AIFInprocessing():
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
    def __init__(self, dataset_train, dataset_test, sens_attrs, privileged_groups,
        unprivileged_groups, label, metric, link, remove):
        self.dataset_train = dataset_train
        self.dataset_test = dataset_test
        self.sens_attrs = sens_attrs
        self.privileged_groups = privileged_groups
        self.unprivileged_groups = unprivileged_groups
        self.label = label
        self.metric = metric
        self.link = link
        self.remove = remove


    def adversarial_debiasing(self, classifier, weight, debias, do_eval=False):
        """
        ...
        """
        tf.compat.v1.disable_eager_execution()
        sess = tf.compat.v1.Session()
        clf = AdversarialDebiasing(self.unprivileged_groups, self.privileged_groups,
            scope_name=classifier, debias=debias, sess=sess, adversary_loss_weight=weight)
        #Requires BinaryLabelDataset
        clf.fit(self.dataset_train)
        pred_dataset = clf.predict(self.dataset_test)
        pred = [item for sublist in pred_dataset.labels.tolist() for item in sublist]
        sess.close()
        tf.compat.v1.reset_default_graph()

        if do_eval:
            metric_val = acc_bias(self.dataset_test, np.asarray(pred).reshape(-1,1), self.unprivileged_groups, self.privileged_groups, self.metric)
        else:
            metric_val = None

        return pred, metric_val, "AdversarialDebiasing", clf


    def gerryfair(self, classifier, gamma, metric="FP", do_eval=False):
        """
        ...
        """
        #SP actually not supported yet
        #metric = "SP"
        #setting predictor=classifier causes issues
        #clf = GerryFairClassifier(gamma=gamma, predictor=classifier, fairness_def=metric)
        clf = GerryFairClassifier(gamma=gamma, fairness_def=metric)
        clf.fit(self.dataset_train)
        pred_dataset = clf.predict(self.dataset_test)
        pred = [item for sublist in pred_dataset.labels.tolist() for item in sublist]

        if do_eval:
            metric_val = acc_bias(self.dataset_test, np.asarray(pred).reshape(-1,1), self.unprivileged_groups, self.privileged_groups, self.metric)
        else:
            metric_val = None

        return pred, metric_val, "GerryFairClassifier", clf


    def metafair(self, tau, do_eval=False):
        """
        ...
        """
        if self.metric == "demographic_parity":
            metric = "sr"
        else:
            metric = "fdr"
        clf = MetaFairClassifier(sensitive_attr=self.sens_attrs[0], type=metric, tau=tau)
        clf.fit(self.dataset_train)
        pred_dataset = clf.predict(self.dataset_test)
        pred = [item for sublist in pred_dataset.labels.tolist() for item in sublist]

        if do_eval:
            metric_val = acc_bias(self.dataset_test, np.asarray(pred).reshape(-1,1), self.unprivileged_groups, self.privileged_groups, self.metric)
        else:
            metric_val = None

        return pred, metric_val, "MetaFairClassifier", clf


    def prejudice_remover(self, eta, do_eval=False):
        """
        ...
        """
        #Only works with one sensitive attribute
        clf = PrejudiceRemover(eta=eta, sensitive_attr=self.sens_attrs[0], class_attr=self.label)
        clf.fit(self.dataset_train)
        pred_dataset = clf.predict(self.dataset_test)
        pred = [item for sublist in pred_dataset.labels.tolist() for item in sublist]

        if do_eval:
            metric_val = acc_bias(self.dataset_test, np.asarray(pred).reshape(-1,1), self.unprivileged_groups, self.privileged_groups, self.metric)
        else:
            metric_val = None

        return pred, metric_val, "PrejudiceRemover", clf


    def exponentiated_gradient_reduction(self, classifier, eps, eta, drop_prot_attr, do_eval=False):
        """
        ...
        """
        if self.metric == "demographic_parity":
            metric = "DemographicParity"
        elif self.metric in ("equalized_odds", "equal_opportunity"):
            metric = "EqualizedOdds"
        else:
            metric = "DemographicParity"

        if self.remove:
            drop_prot_attr = True
        clf = ExponentiatedGradientReduction(classifier, metric, eps=eps, eta0=eta, drop_prot_attr=drop_prot_attr)
        clf.fit(self.dataset_train)
        pred_dataset = clf.predict(self.dataset_test)
        pred = [item for sublist in pred_dataset.labels.tolist() for item in sublist]

        if do_eval:
            metric_val = acc_bias(self.dataset_test, np.asarray(pred).reshape(-1,1), self.unprivileged_groups, self.privileged_groups, self.metric)
        else:
            metric_val = None

        return pred, metric_val, "ExponentiatedGradientReduction", clf


    def gridsearch_reduction(self, classifier, weight, drop_prot_attr, do_eval=False):
        """
        ...
        """
        if self.metric == "demographic_parity":
            metric = "DemographicParity"
        elif self.metric in ("equalized_odds", "equal_opportunity"):
            metric = "EqualizedOdds"
        else:
            metric = "DemographicParity"

        if self.remove:
            drop_prot_attr = True
        clf = GridSearchReduction(classifier, metric, self.sens_attrs, constraint_weight=weight, drop_prot_attr=drop_prot_attr)
        clf.fit(self.dataset_train)
        pred_dataset = clf.predict(self.dataset_test)
        pred = [item for sublist in pred_dataset.labels.tolist() for item in sublist]

        if do_eval:
            metric_val = acc_bias(self.dataset_test, np.asarray(pred).reshape(-1,1), self.unprivileged_groups, self.privileged_groups, self.metric)
        else:
            metric_val = None

        return pred, metric_val, "GridSearchReduction", clf

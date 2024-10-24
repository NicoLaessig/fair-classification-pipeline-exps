"""
In this python file multiple classification models are trained.
"""
import numpy as np
import pandas as pd
import copy
import os
import tensorflow as tf
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from fairlearn.reductions import ExponentiatedGradient, GridSearch
from fairlearn.reductions import DemographicParity, ErrorRate, EqualizedOdds
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from .AdaFair.AdaFair import AdaFair
from .AdaFair.AdaFairEQOP import AdaFairEQOP
from .AdaFair.AdaFairSP import AdaFairSP
from .fair_glm_cvx.models import *
from .FAGTB.functions import FAGTB
from .adv_deb_multi.adversarial_debiasing_multi import AdversarialDebiasingMulti
from .gradual_compatibility.lr import CustomLogisticRegression
from .fair_dummies.fair_dummies_learning import EquiClassLearner
from .fair_dummies.others.continuous_fairness import HGR_Class_Learner
from .fairret.loss import NormLoss
from .fairret.statistic import FalsePositiveRate, TruePositiveRate, FalseNegativeFalsePositiveFraction, StackedLinearFractionalStatistic, PositiveRate
from .evaluation.eval_classifier import acc_bias


class Inprocessing():
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
    def __init__(self, X_train, X_test, y_train, y_test, sens_attrs, favored,
        dataset_train, dataset_test, privileged_groups, unprivileged_groups, label,
        metric, link, remove):
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test
        self.sens_attrs = sens_attrs
        self.favored = favored
        self.dataset_train = dataset_train
        self.dataset_test = dataset_test
        self.privileged_groups = privileged_groups
        self.unprivileged_groups = unprivileged_groups
        self.label = label
        self.metric = metric
        self.X_tr_np = self.X_train.to_numpy()
        self.y_tr_np = self.y_train.to_numpy().reshape((self.y_train.to_numpy().shape[0],))
        self.X_te_np = self.X_test.to_numpy()
        self.y_te_np = self.y_test.to_numpy().reshape((self.y_test.to_numpy().shape[0],))
        self.link = link
        self.remove = remove


    def adafair(self, classifier, iterations=10, learning_rate=1, do_eval=False):
        sens_idx = self.X_train.columns.get_loc(self.sens_attrs[0])
        if self.metric == "demographic_parity":
            clf = AdaFairSP(base_estimator=classifier, n_estimators=iterations, learning_rate=learning_rate, saIndex=sens_idx, saValue=self.favored)
        elif self.metric in ("equalized_odds", "equal_opportunity"):
            clf = AdaFairEQOP(base_estimator=classifier, n_estimators=iterations, learning_rate=learning_rate, saIndex=sens_idx, saValue=self.favored)
        else:
            clf = AdaFair(base_estimator=classifier, n_estimators=iterations, learning_rate=learning_rate, saIndex=sens_idx, saValue=self.favored)

        clf.fit(self.X_train, self.y_train)
        pred = clf.predict(self.X_test)

        if do_eval:
            metric_val = acc_bias(self.dataset_test, np.asarray(pred).reshape(-1,1), self.unprivileged_groups, self.privileged_groups, self.metric)
        else:
            metric_val = None

        return pred, metric_val, "AdaFair", clf


    def fglm(self, lam, family, discretization, do_eval=False):
        sens_idx = self.X_train.columns.get_loc(self.sens_attrs[0])

        clf = FairGeneralizedLinearModel(sensitive_index=sens_idx, lam=lam, family=family, discretization=discretization)

        clf.fit(self.X_train.to_numpy(), self.y_train.to_numpy())
        pred = clf._predict(self.X_te_np)

        if do_eval:
            metric_val = acc_bias(self.dataset_test, np.asarray(pred).reshape(-1,1), self.unprivileged_groups, self.privileged_groups, self.metric)
        else:
            metric_val = None

        return pred, metric_val, "FairGeneralizedLinearModel", clf


    def squared_diff_fair_logistic(self, lam, do_eval=False):
        sens_idx = self.X_train.columns.get_loc(self.sens_attrs[0])

        clf = SquaredDifferenceFairLogistic(sensitive_index=sens_idx, lam=lam)

        clf.fit(self.X_tr_np, self.y_tr_np)
        pred = clf._predict(self.X_te_np)

        if do_eval:
            metric_val = acc_bias(self.dataset_test, np.asarray(pred).reshape(-1,1), self.unprivileged_groups, self.privileged_groups, self.metric)
        else:
            metric_val = None

        return pred, metric_val, "SquaredDifferenceFairLogistic", clf


    def fairness_constraint_model(self, c, tau, mu, eps, do_eval=False):
        sens_idx = self.X_train.columns.get_loc(self.sens_attrs[0])

        clf = FairnessConstraintModel(sensitive_index=sens_idx, c=c, tau=tau, mu=mu, eps=eps)
        clf.fit(self.X_tr_np, self.y_tr_np)
        pred = clf.predict(self.X_te_np)

        if do_eval:
            metric_val = acc_bias(self.dataset_test, np.asarray(pred).reshape(-1,1), self.unprivileged_groups, self.privileged_groups, self.metric)
        else:
            metric_val = None

        return pred, metric_val, "FairnessConstraintModel", clf


    def disparate_treatment_model(self, c, tau, mu, eps, do_eval=False):
        sens_idx = self.X_train.columns.get_loc(self.sens_attrs[0])

        clf = DisparateMistreatmentModel(sensitive_index=sens_idx, c=c, tau=tau, mu=mu, eps=eps)

        clf.fit(self.X_tr_np, self.y_tr_np)
        pred = clf._predict(self.X_te_np)

        if do_eval:
            metric_val = acc_bias(self.dataset_test, np.asarray(pred).reshape(-1,1), self.unprivileged_groups, self.privileged_groups, self.metric)
        else:
            metric_val = None

        return pred, metric_val, "DisparateMistreatmentModel", clf


    def convex_framework(self, lam, family, penalty, do_eval=False):
        #INDIVIDUAL & GROUP FAIRNESS PENALTY OPTION
        sens_idx = self.X_train.columns.get_loc(self.sens_attrs[0])

        clf = ConvexFrameworkModel(sensitive_index=sens_idx, lam=lam, family=family, penalty=penalty)

        clf.fit(self.X_tr_np, self.y_tr_np)
        pred = clf._predict(self.X_te_np)

        if family == "normal":
            for i, pr in enumerate(pred):
                if pr >= 0.5:
                    pred[i] = 1
                else:
                    pred[i] = 0

        if do_eval:
            metric_val = acc_bias(self.dataset_test, np.asarray(pred).reshape(-1,1), self.unprivileged_groups, self.privileged_groups, self.metric)
        else:
            metric_val = None

        return pred, metric_val, "ConvexFrameworkModel", clf


    def hsic_linear_regression(self, lam, do_eval=False):
        sens_idx = self.X_train.columns.get_loc(self.sens_attrs[0])

        clf = HSICLinearRegression(sensitive_index=sens_idx, lam=lam)

        clf.fit(self.X_tr_np, self.y_tr_np)
        pred = clf._predict(self.X_te_np)

        for i in range(len(pred)):
            if pred[i] < 0.5:
                pred[i] = 0
            else:
                pred[i] = 1

        if do_eval:
            metric_val = acc_bias(self.dataset_test, np.asarray(pred).reshape(-1,1), self.unprivileged_groups, self.privileged_groups, self.metric)
        else:
            metric_val = None

        return pred, metric_val, "HSICLinearRegression", clf


    def general_ferm(self, eps, k, do_eval=False):
        sens_idx = self.X_train.columns.get_loc(self.sens_attrs[0])

        clf = GeneralFairERM(sensitive_index=sens_idx, eps=eps, K=k)

        clf.fit(self.X_tr_np, self.y_tr_np)
        pred = clf._predict(self.X_te_np)

        for i in range(len(pred)):
            if pred[i] < 0:
                pred[i] = 0
            else:
                pred[i] = 1

        if do_eval:
            metric_val = acc_bias(self.dataset_test, np.asarray(pred).reshape(-1,1), self.unprivileged_groups, self.privileged_groups, self.metric)
        else:
            metric_val = None

        return pred, metric_val, "GeneralFairERM", clf


    def fagtb(self, estimators, learning, lam, do_eval=False):
        sens_idx = self.X_train.columns.get_loc(self.sens_attrs[0])
        X_train = copy.deepcopy(self.X_train)
        X_test = copy.deepcopy(self.X_test)
        if self.remove: 
            for sens in self.sens_attrs:
                X_train = X_train.drop(sens, axis=1)
                X_test = X_test.drop(sens, axis=1)

        #Only first 3 parameters are actually used
        clf = FAGTB(n_estimators=estimators, learning_rate=learning, max_features=None,
            min_samples_split=2, min_impurity=None, max_depth=9, regression=None)

        clf.fit(X_train.to_numpy(), self.y_tr_np, sensitive=self.X_train[self.sens_attrs[0]].values, LAMBDA=lam, Xtest=X_test.to_numpy(),
            yt=self.y_te_np, sensitivet=self.X_test[self.sens_attrs[0]].values)
        #How are the predictions returned?
        pred = clf.predict(X_test.to_numpy())

        for i, p in enumerate(pred):
            if p >= 0.5:
                pred[i] = 1
            else:
                pred[i] = 0

        if do_eval:
            metric_val = acc_bias(self.dataset_test, np.asarray(pred).reshape(-1,1), self.unprivileged_groups, self.privileged_groups, self.metric)
        else:
            metric_val = None

        return pred, metric_val, "FAGTB", clf


    def fair_dummies(self, batch=32, lr=0.5, mu=0.99999, second_scale=0.01, epochs=50, model_type="linear_model", do_eval=False):
        X_train = self.X_train.drop(self.sens_attrs, axis=1).values
        X_test = self.X_test.drop(self.sens_attrs, axis=1).values
        A_train = self.X_train[self.sens_attrs[0]]
        A_test = self.X_test[self.sens_attrs[0]]
        clf = EquiClassLearner(lr=lr, pretrain_pred_epochs=2, pretrain_dis_epochs=2, epochs=epochs,
            loss_steps=1, dis_steps=1, cost_pred=torch.nn.CrossEntropyLoss(),
            in_shape=X_train.shape[1], batch_size=batch, model_type=model_type, lambda_vec=mu,
            second_moment_scaling=second_scale, num_classes=2)

        input_data_train = np.concatenate((A_train[:,np.newaxis], X_train), 1)
        input_data_test = np.concatenate((A_test[:,np.newaxis], X_test), 1)

        clf.fit(input_data_train, self.y_tr_np)
        prediction_list = clf.predict(input_data_test)
        pred = []
        for i, prediction in enumerate(prediction_list):
            if prediction_list[i][0] > prediction_list[i][1]:
                pred.append(0)
            else:
                pred.append(1)

        if do_eval:
            metric_val = acc_bias(self.dataset_test, np.asarray(pred).reshape(-1,1), self.unprivileged_groups, self.privileged_groups, self.metric)
        else:
            metric_val = None

        return pred, metric_val, "FairDummies", clf


    def hgr(self, batch=128, lr=0.001, mu=0.98, epochs=50, model_type="linear_model", do_eval=False):
        size_check = (len(self.X_train)-1)/32
        if size_check.is_integer():
            X_train = self.X_train[:-1]
            y_train = self.y_train[:-1]
        else:
            X_train = copy.deepcopy(self.X_train)
            y_train = copy.deepcopy(self.y_train)
        A_train = X_train[self.sens_attrs[0]]
        A_test = self.X_test[self.sens_attrs[0]]
        X_train = X_train.drop(self.sens_attrs[0], axis=1).values
        X_test = self.X_test.drop(self.sens_attrs[0], axis=1).values
        
        clf = HGR_Class_Learner(lr=lr, epochs=epochs, mu=mu, cost_pred=torch.nn.CrossEntropyLoss(),
            in_shape=X_train.shape[1], out_shape=2, batch_size=batch, model_type=model_type)
        input_data_train = np.concatenate((A_train[:,np.newaxis], X_train), 1)
        input_data_test = np.concatenate((A_test[:,np.newaxis], X_test), 1)
        y_tr_np = y_train.to_numpy().reshape((y_train.to_numpy().shape[0],))
        
        clf.fit(input_data_train, y_tr_np)
        prediction_list = clf.predict(input_data_test)
        pred = []
        for i, prediction in enumerate(prediction_list):
            if prediction_list[i][0] > prediction_list[i][1]:
                pred.append(0)
            else:
                pred.append(1)

        if do_eval:
            metric_val = acc_bias(self.dataset_test, np.asarray(pred).reshape(-1,1), self.unprivileged_groups, self.privileged_groups, self.metric)
        else:
            metric_val = None

        return pred, metric_val, "HGR", clf


    def multi_adv_deb(self, weight, do_eval=False):
        if self.metric == "demographic_parity":
            fairness_def = "parity"
        elif self.metric in ("equalized_odds", "equal_opportunity"):
            fairness_def = "equal_odds"
        else:
            fairness_def = "parity"

        df_train = pd.merge(self.X_train, self.y_train, left_index=True, right_index=True)
        meta_train = df_train[[self.sens_attrs[0], self.label]]
        features_train = df_train.drop(columns=[self.sens_attrs[0], self.label])
        df_test = pd.merge(self.X_test, self.y_test, left_index=True, right_index=True)
        meta_test = df_test[[self.sens_attrs[0], self.label]]
        features_test = df_test.drop(columns=[self.sens_attrs[0], self.label])

        meta_train['label'] = meta_train[self.label].astype(int)
        meta_test['label'] = meta_test[self.label].astype(int)
        meta_train.reset_index(drop=True, inplace=True)
        meta_test.reset_index(drop=True, inplace=True)
        features_train.reset_index(drop=True, inplace=True)
        features_test.reset_index(drop=True, inplace=True)

        sess = tf.compat.v1.Session()
        tf.compat.v1.disable_eager_execution()

        model = AdversarialDebiasingMulti(
            protected_attribute_name=self.sens_attrs[0],
            num_labels=len(meta_train[self.label].unique()),
            scope_name='biased_classifier',
            debias=True,
            adversary_loss_weight=weight,
            fairness_def=fairness_def,
            verbose=False,
            num_epochs=64,
            classifier_num_hidden_units_1=60,
            classifier_num_hidden_units_2=20,
            sess=sess
        )
        model.fit(features_train, meta_train)
        pred = model.predict(features_test, meta_test)["pred_label"].to_numpy()
        sess.close()
        tf.compat.v1.reset_default_graph()

        if do_eval:
            metric_val = acc_bias(self.dataset_test, np.asarray(pred).reshape(-1,1), self.unprivileged_groups, self.privileged_groups, self.metric)
        else:
            metric_val = None

        return pred, metric_val, "MultiAdversarialDebiasing", model


    def grad_compat(self, reg=0, reg_val=1, weights_init=None, lambda_=0, do_eval=False):
        if weights_init == "None":
            weights_init = None

        X_train = self.X_train.drop(self.sens_attrs, axis=1).values
        X_test = self.X_test.drop(self.sens_attrs, axis=1).values
        A_train = self.X_train[self.sens_attrs].values
        A_test = self.X_test[self.sens_attrs].values

        alpha = 0
        beta = 0
        gamma = 0
        if reg in (1, 4, 5, 7):
            alpha = reg_val
        elif reg in (2, 4, 6, 7):
            beta = reg_val
        elif reg in (3, 5, 6, 7):
            gamma = reg_val

        model = CustomLogisticRegression(X=X_train, Y=self.y_tr_np, A=A_train.reshape(-1,),
            weights_init=weights_init, alpha=alpha, beta=beta, gamma=gamma, _lambda=lambda_)
        model.fit()
        
        pred = []
        prediction = model.predict_prob(X_test)
        for i in range(len(prediction)):
            pred.append(1) if prediction[i] > 0.5 else pred.append(0)

        if do_eval:
            metric_val = acc_bias(self.dataset_test, np.asarray(pred).reshape(-1,1), self.unprivileged_groups, self.privileged_groups, self.metric)
        else:
            metric_val = None

        return pred, metric_val, "GradualCompatibility", model


    def fairret(self, lam, lr, h_layer, do_eval=False):
        batch_size = 4096
        nb_epochs = 100

        feat = self.X_train.drop(columns=self.sens_attrs).to_numpy(dtype="float")
        sens = self.X_train[self.sens_attrs].to_numpy(dtype="float")
        label = self.y_train.to_numpy(dtype="float")

        test = self.X_test.drop(columns=self.sens_attrs).to_numpy(dtype="float")
        test = torch.tensor(test).float()

        feat, sens, label = torch.tensor(feat).float(), torch.tensor(sens).float(), torch.tensor(label).float()

        dataset = TensorDataset(feat, sens, label)
        dataloader = DataLoader(dataset, batch_size=batch_size)

        model = nn.Sequential(
            nn.Linear(feat.shape[1], h_layer),
            nn.ReLU(),
            nn.Linear(h_layer, 1)
        )

        if self.metric == "equal_opportunity":
            statistic = TruePositiveRate()
        elif self.metric == "equalized_odds":
            statistic = StackedLinearFractionalStatistic(TruePositiveRate(), FalsePositiveRate())
        elif self.metric == "treatment_equality":
            statistic = FalseNegativeFalsePositiveFraction()
        else:
            statistic = PositiveRate()

        norm_loss = NormLoss(statistic)

        optimizer = optim.Adam(model.parameters(), lr=lr)

        for epoch in range(nb_epochs):
            losses = []
            for batch_feat, batch_sens, batch_label in dataloader:
                optimizer.zero_grad()
                        
                logit = model(batch_feat)
                loss = nn.functional.binary_cross_entropy_with_logits(logit, batch_label)

                if epoch >= 20:
                    if self.metric in ("demographic_parity", "consistency"):
                        loss += lam * norm_loss(logit, batch_sens)
                    else:   
                        loss += lam * norm_loss(logit, batch_sens, batch_label)
                loss.backward()
                        
                optimizer.step()
                losses.append(loss.item())

        pred_list = torch.sigmoid(model(test)).detach().numpy()
        pred = []
        for p in pred_list:
            if p >= 0.5:
                pred.append(1)
            else:
                pred.append(0)

        if do_eval:
            metric_val = acc_bias(self.dataset_test, np.asarray(pred).reshape(-1,1), self.unprivileged_groups, self.privileged_groups, self.metric)
        else:
            metric_val = None

        return pred, metric_val, "fairret", model

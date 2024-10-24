"""
In this python file multiple classification models are trained.
"""
import math
import copy
import time
import numpy as np
import pandas as pd
import torch
import random
from sklearn.neighbors import kneighbors_graph
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.model_selection import train_test_split
from sklearn.semi_supervised import SelfTrainingClassifier, LabelSpreading, LabelPropagation
from sklearn.utils import shuffle
from sklearn.linear_model import LogisticRegression
from scipy import stats
from .Fair_SMOTE.SMOTE import smote
from .Fair_SMOTE.Generate_Samples import generate_samples
from .FESF.fesf import FESF
from .iFlipper import *
from .PFR.PFR import *
from .evaluation.eval_classifier import acc_bias


class Preprocessing():
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
    def __init__(self, X_train, X_test, y_train, y_test, sens_attrs, favored, dataset_train,
        dataset_test, testsize, randomstate, privileged_groups, unprivileged_groups,
        index, label, metric, combine, link, input_file, remove):
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test
        self.sens_attrs = sens_attrs
        self.favored = favored
        self.dataset_train = dataset_train
        self.dataset_test = dataset_test
        self.testsize = testsize
        self.randomstate = randomstate
        self.privileged_groups = privileged_groups
        self.unprivileged_groups = unprivileged_groups
        self.index = index
        self.label = label
        self.metric = metric
        self.combine = combine
        self.X_tr_np = self.X_train.to_numpy()
        self.y_tr_np = self.y_train.to_numpy().reshape((self.y_train.to_numpy().shape[0],))
        self.X_te_np = self.X_test.to_numpy()
        self.y_te_np = self.y_test.to_numpy().reshape((self.y_test.to_numpy().shape[0],))
        self.link = link
        self.input_file = input_file
        self.remove = remove


    def smote(self, classifier, cr, f, do_eval=False):
        label = list(self.y_train.columns)[0]
        train_df = copy.deepcopy(self.X_train)
        train_df[label] = copy.deepcopy(self.y_train)
        dataset_orig_train = copy.deepcopy(train_df)

        dict_cols = dict()
        cols = list(dataset_orig_train.columns)
        for i, col in enumerate(cols):
            dict_cols[i] = col

        lists = [[0, 1]]
        for sens in self.sens_attrs:
            sens_vals = []
            for val in pd.unique(self.X_train[[sens]].values.ravel()):
                sens_vals.append(val)
            lists.append(sens_vals)

        groups = [label] + self.sens_attrs
        grouped_train = dataset_orig_train.groupby(groups)
        groups_length = []
        for key, value in grouped_train:
            groups_length.append(len(grouped_train.get_group(key)))

        max_val = max(groups_length)
        count = 0
        for key, value in grouped_train:
            gdf = grouped_train.get_group(key)
            if len(gdf) < max_val:
                increase = max_val - len(gdf)
                gdf_bal = generate_samples(increase,gdf,'',dict_cols,cr,f)
            else:
                gdf_bal = copy.deepcopy(gdf)
            if count == 0:
                df_new = copy.deepcopy(gdf_bal)
            else:
                df_new = df_new.append(gdf_bal)
            count += 1

        X_train, y_train = df_new.loc[:, df_new.columns != label], df_new[label]
        X_test = copy.deepcopy(self.X_test)
        if self.remove:
            for sens in self.sens_attrs:
                X_train = X_train.drop(sens, axis=1)
                X_test = X_test.drop(sens, axis=1)

        classifier.fit(X_train, y_train)
        pred = classifier.predict(X_test)

        if do_eval:
            metric_val = acc_bias(self.dataset_test, np.asarray(pred).reshape(-1,1), self.unprivileged_groups, self.privileged_groups, self.metric)
        else:
            metric_val = None

        return pred, X_train, y_train, metric_val, "Fair-SMOTE", classifier


    def fairssl(self, classifier, ssl_type, balancing, cr, f, do_eval=False):
        label = list(self.y_train.columns)[0]
        train_df = copy.deepcopy(self.X_train)
        train_df[label] = copy.deepcopy(self.y_train)
        dataset_orig_train = copy.deepcopy(train_df)

        grouped_train = dataset_orig_train.groupby(self.sens_attrs)
        clfs = []
        protected = None
        unprotected = None
        for key, value in grouped_train:
            gdf = grouped_train.get_group(key)
            X_train_group, y_train_group = gdf.loc[:, gdf.columns != label], gdf[label]
            clf_group = copy.deepcopy(classifier)
            clf_group.fit(X_train_group, y_train_group)
            clfs.append(clf_group)
            if key == self.favored:
                if unprotected is None:
                    X_train_unprotected = copy.deepcopy(X_train_group)
                    y_train_unprotected = copy.deepcopy(y_train_group)
                else:
                    X_train_unprotected.append(X_train_group)
                    y_train_unprotected.append(y_train_group)
            else:
                if protected is None:
                    X_train_protected = copy.deepcopy(X_train_group)
                    y_train_protected = copy.deepcopy(y_train_group)
                else:
                    X_train_protected.append(X_train_group)
                    y_train_protected.append(y_train_group)
        clf_unprotected = copy.deepcopy(classifier)
        clf_unprotected.fit(X_train_unprotected, y_train_unprotected)
        clf_protected = copy.deepcopy(classifier)
        clf_protected.fit(X_train_protected, y_train_protected)

        unlabeled_df = pd.DataFrame(columns=dataset_orig_train.columns)

        count = 0
        for key, value in grouped_train:
            gdf = grouped_train.get_group(key)
            if key == self.favored:
                y_same = clfs[count].predict(self.X_train.loc[list(gdf.index)])
                y_other = clf_protected.predict(self.X_train.loc[list(gdf.index)])
            else:
                y_same = clfs[count].predict(self.X_train.loc[list(gdf.index)])
                y_other = clf_unprotected.predict(self.X_train.loc[list(gdf.index)])

            ic = 0
            for i, row in gdf.iterrows():
                if y_same[ic] != y_other[ic]:
                    unlabeled_df = unlabeled_df.append(dataset_orig_train.loc[i], ignore_index=True)
                    dataset_orig_train = dataset_orig_train.drop(i)
                ic += 1
            count += 1

        dict_cols = dict()
        cols = list(train_df.columns)
        for i, col in enumerate(cols):
            dict_cols[i] = col

        x = int(len(self.X_train)/25)

        count = 0
        grouped_train = dataset_orig_train.groupby(self.sens_attrs)
        for key, value in grouped_train:
            gdf = grouped_train.get_group(key)
            if count == 0:
                labeled_df = dataset_orig_train.loc[list(gdf.index)][:x]
            else:
                labeled_df.append(dataset_orig_train.loc[list(gdf.index)][:x])
            unlabeled_df.append(dataset_orig_train.loc[list(gdf.index)][x:])
            count += 1

        labeled_df = shuffle(labeled_df.reset_index(drop=True))
        unlabeled_df = unlabeled_df.reset_index(drop=True)

        unlabeled_df[label] = -1
        mixed_df = labeled_df.append(unlabeled_df)

        X_train, y_train = mixed_df.loc[:, mixed_df.columns != label], mixed_df[label]

        if ssl_type in ("SelfTraining", "LabelSpreading", "LabelPropagation"):
            if ssl_type == "SelfTraining":
                model_name = "FairSSL-ST"
                training_model = SelfTrainingClassifier(classifier)
                training_model.fit(X_train, y_train)
            elif ssl_type == "LabelSpreading":
                model_name = "FairSSL-LS"
                training_model = LabelSpreading()
                training_model.fit(X_train, y_train)
            elif ssl_type == "LabelPropagation":
                model_name = "FairSSL-LP"
                training_model = LabelPropagation()
                training_model.fit(X_train, y_train)
            X_unl, y_unl = unlabeled_df.loc[:, unlabeled_df.columns != label], unlabeled_df[label]
            y_pred_proba = training_model.predict_proba(X_unl)
            y_pred = training_model.predict(X_unl)

            to_keep = []

            for i in range(len(y_pred_proba)):
                if max(y_pred_proba[i]) >= 0.6:
                    to_keep.append(i)

            X_unl_certain = X_unl.iloc[to_keep,:]
            y_unl_certain = y_pred[to_keep]

            X_train, y_train = labeled_df.loc[:, labeled_df.columns != label], labeled_df[label]

            X_train = X_train.append(X_unl_certain)
            y_train = np.concatenate([y_train, y_unl_certain])
        elif ssl_type == "CoTraining":
            model_name = "FairSSL-CT"
            X_unl, y_unl = unlabeled_df.loc[:, unlabeled_df.columns != label], unlabeled_df[label]
            X_train, y_train = labeled_df.loc[:, labeled_df.columns != label], labeled_df[label]
            classifier_dict = {}
            for i in X_train.columns:
                classifier_dict[i] = classifier.fit(X_train[[i]],y_train)
            length_of_columns = len(X_unl.columns)
            change = 0
            for index, row in X_unl.iterrows():
                try:
                    prediction_for_current_point = 0
                    for j in classifier_dict:
                        prediction_for_current_point += classifier_dict[j].predict([[row[j]]])[0]
                    if prediction_for_current_point > (length_of_columns//2) + 1:
                        y_unl.loc[index] = 1
                        change += 1
                    elif prediction_for_current_point < (length_of_columns//2) - 1:
                        y_unl.loc[index] = 0
                        change += 1
                    else:
                        X_unl = X_unl.drop(index)
                        y_unl = y_unl.drop(index)
                except:
                    break
            X_train = X_train.append(X_unl)
            y_train = y_train.append(y_unl)

        if balancing:
            dataset_train = copy.deepcopy(X_train)
            dataset_train[label] = y_train
            groups = [label] + self.sens_attrs
            grouped_train = dataset_train.groupby(groups)
            groups_length = []
            for key, value in grouped_train:
                groups_length.append(len(grouped_train.get_group(key)))

            max_val = max(groups_length)
            count = 0
            for key, value in grouped_train:
                gdf = grouped_train.get_group(key)
                if len(gdf) < max_val:
                    increase = max_val - len(gdf)
                    gdf_bal = generate_samples(increase,gdf,'',dict_cols,cr,f)
                else:
                    gdf_bal = copy.deepcopy(gdf)
                if count == 0:
                    df_new = copy.deepcopy(gdf_bal)
                else:
                    df_new = df_new.append(gdf_bal)
                count += 1
            X_train, y_train = df_new.loc[:, df_new.columns != label], df_new[label]
        
        X_test = copy.deepcopy(self.X_test)
        if self.remove:
            for sens in self.sens_attrs:
                X_train = X_train.drop(sens, axis=1)
                X_test = X_test.drop(sens, axis=1)

        if isinstance(y_train, np.ndarray):
            y_train = pd.Series(y_train, name=self.label)


        classifier.fit(X_train, y_train)
        pred = classifier.predict(X_test)

        if do_eval:
            metric_val = acc_bias(self.dataset_test, np.asarray(pred).reshape(-1,1), self.unprivileged_groups, self.privileged_groups, self.metric)
        else:
            metric_val = None


        return pred, X_train, y_train, metric_val, model_name, classifier


    def ltdd(self, classifier, do_eval=False):
        """...
        """
        X_train = copy.deepcopy(self.X_train)
        X_test = copy.deepcopy(self.X_test)

        column_train = [column for column in X_train]
        ce_list = []
        ce_times = []
        slope_store = []
        intercept_store = []
        rvalue_store = []
        pvalue_store = []
        column_u = []
        flag = 0
        ce = []
        times = 0

        def Linear_regression(x, slope, intercept):
            return x * slope + intercept

        for i in column_train:
            flag = flag + 1
            if i != self.sens_attrs[0]:
                slope, intercept, rvalue, pvalue, stderr = stats.linregress(X_train[self.sens_attrs[0]], X_train[i])
                rvalue_store.append(rvalue)
                pvalue_store.append(pvalue)
                if pvalue < 0.05:
                    times = times + 1
                    column_u.append(i)
                    ce.append(flag)
                    slope_store.append(slope)
                    intercept_store.append(intercept)
                    X_train[i] = X_train[i] - Linear_regression(X_train[self.sens_attrs[0]], slope, intercept)

        ce_times.append(times)
        ce_list.append(ce)

        X_train2 = X_train.drop([self.sens_attrs[0]],axis = 1)

        for i in range(len(column_u)):
            X_test[column_u[i]] = X_test[column_u[i]] - Linear_regression(X_test[self.sens_attrs[0]], slope_store[i],
                intercept_store[i])

        X_test2 = X_test.drop([self.sens_attrs[0]],axis = 1)

        classifier.fit(X_train2, self.y_train)
        pred = classifier.predict(X_test2)

        if do_eval:
            metric_val = acc_bias(self.dataset_test, np.asarray(pred).reshape(-1,1), self.unprivileged_groups, self.privileged_groups, self.metric)
        else:
            metric_val = None

        return pred, X_train, self.y_train, X_test, metric_val, "LTDD", classifier


    def fair_rr(self, classifier, level, do_eval=False):
        """
        ...
        """
        def Find_Theta(t,p11,p10,p01,p00,fairness):
            if t >= 0:
                if fairness == 'DP':
                    theta11 = t / (p11 + p10 + t)
                    theta00 = t / (p01 + p00 + t)
                    theta10 = 0
                    theta01 = 0
                if fairness == 'EO':
                    theta11 = t / 2 / p11
                    theta00 = t / 2 / (p01 + t)
                    theta10 = 0
                    theta01 = 0
                if fairness == 'PE':
                    theta11 = t / 2 / (p10 + t)
                    theta00 = t / 2 / p00
                    theta10 = 0
                    theta01 = 0
            if t < 0:
                if fairness == 'DP':
                    theta11 = 0
                    theta00 = 0
                    theta10 = t / (t - p11 - p10)
                    theta01 = t / (t - p01 - p00)
                if fairness == 'EO':
                    theta11 = 0
                    theta00 = 0
                    theta10 = t / 2 / (t - p11)
                    theta01 = -1 * t / 2 / p01
                if fairness == 'PE':
                    theta11 = 0
                    theta00 = 0
                    theta10 = -1 * t / 2 / p10
                    theta01 = t / 2 / (t - p00)

            return theta11, theta10, theta01, theta00

        def test_all_parity(fairness, Y_test, Z_test, pred):
            if fairness == 'DP':
                parity = pred[Z_test == 1].mean() - pred[Z_test == 0].mean()
            elif fairness == 'EO':
                parity = pred[(Z_test == 1) & (Y_test == 1)].mean() - pred[(Z_test == 0) & (Y_test == 1)].mean()
            elif fairness == 'PE':
                parity = pred[(Z_test == 1) & (Y_test == 0)].mean() - pred[(Z_test == 0) & (Y_test == 0)].mean()

            return parity


        if self.metric in ("equalized_odds", "equal_opportunity"):
            fairness = "EO"
        elif self.metric in ("treatment_equality"):
            fairness = "PE"
        else:
            fairness = "DP"

        label = list(self.y_train.columns)[0]
        training_set = copy.deepcopy(self.X_train)
        training_set[label] = copy.deepcopy(self.y_train)

        Z_train = training_set[self.sens_attrs[0]]
        Y_train = training_set[label]
        X_train = training_set.drop(label, axis=1)

        training_set11 = training_set[(training_set[self.sens_attrs[0]] == 1) & (training_set[label] == 1)]
        training_set10 = training_set[(training_set[self.sens_attrs[0]] == 1) & (training_set[label] == 0)]
        training_set01 = training_set[(training_set[self.sens_attrs[0]] == 0) & (training_set[label] == 1)]
        training_set00 = training_set[(training_set[self.sens_attrs[0]] == 0) & (training_set[label] == 0)]

        n11 = len(training_set11)
        n10 = len(training_set10)
        n01 = len(training_set01)
        n00 = len(training_set00)
        n = len(training_set)
        p11 = n11 / n
        p10 = n10 / n
        p01 = n01 / n
        p00 = n00 / n
        
        training_set11_s = shuffle(training_set11)
        training_set10_s = shuffle(training_set10)
        training_set01_s = shuffle(training_set01)
        training_set00_s = shuffle(training_set00)

        X_test = copy.deepcopy(self.X_test)
        if self.remove:
            for sens in self.sens_attrs:
                X_train = X_train.drop(sens, axis=1)
                X_test = X_test.drop(sens, axis=1)

        classifier.fit(X_train, Y_train)
        train_pred = classifier.predict(X_train)
        #disparity_val = acc_bias(self.dataset_train, np.asarray(train_pred).reshape(-1,1), self.unprivileged_groups, self.privileged_groups, self.metric, lam=0, no_abs=True)
        disparity_val = test_all_parity(fairness, Y_train.to_numpy(), Z_train.to_numpy(), train_pred)
        if abs(disparity_val) <= level:
            pred = classifier.predict(X_test)
            X_syn = X_train
            Y_syn = Y_train
        else:
            if disparity_val > level:
                t_max = p11 + p10
                t_min = 0
                level0 = level
            else:
                t_max = 0
                t_min = -p01 - p00
                level0 = -1 * level
            t_mid = (t_max + t_min) / 2

            while abs(t_max - t_min) > 1e-4:
                X_train11 = training_set11_s.copy()
                X_train10 = training_set10_s.copy()
                X_train01 = training_set01_s.copy()
                X_train00 = training_set00_s.copy()

                theta = Find_Theta(t_mid,p11,p10,p01,p00,fairness)
                num_change11 =  int(n11 * theta[0])
                num_change10 =  int(n10 * theta[1])
                num_change01 =  int(n01 * theta[2])
                num_change00 =  int(n00 * theta[3])

                X_train11[label].values[:num_change11] = 0
                X_train10[label].values[:num_change10] = 1
                X_train01[label].values[:num_change01] = 0
                X_train00[label].values[:num_change00] = 1

                X_syn = pd.concat([X_train11,X_train10,X_train01,X_train00])
                Y_syn = X_syn[label]
                Z_syn = X_syn[self.sens_attrs[0]]
                X_syn = X_syn.drop(label,axis =1)

                if self.remove:
                    for sens in self.sens_attrs:
                        X_syn = X_syn.drop(sens, axis=1)

                classifier.fit(X_syn, Y_syn)
                train_pred = classifier.predict(X_train)

                #disparity_val = acc_bias(self.dataset_train, np.asarray(train_pred).reshape(-1,1), self.unprivileged_groups, self.privileged_groups, self.metric, lam=0, no_abs=True)
                disparity_val = test_all_parity(fairness, Y_train.to_numpy(), Z_train.to_numpy(), train_pred)
                print(disparity_val)
                if disparity_val > level0:
                    t_min = t_min / 3 + 2 * t_mid / 3
                else:
                    t_max = t_max / 3 + 2 * t_mid / 3

                t_mid = (t_max + t_min) / 2


            pred = classifier.predict(X_test)

        if do_eval:
            metric_val = acc_bias(self.dataset_test, np.asarray(pred).reshape(-1,1), self.unprivileged_groups, self.privileged_groups, self.metric)
        else:
            metric_val = None

        return pred, X_syn, Y_syn, metric_val, "FairRR", classifier


    def fesf(self, K, do_eval=False):
        model = FESF(K=K)
        X_train, X_val, y_train2, y_val2 = train_test_split(self.X_train, self.y_train, test_size=0.2, random_state=42)

        S_train2 = X_train[self.sens_attrs[0]]
        S_train = S_train2.to_numpy().reshape((S_train2.to_numpy().shape[0],))
        S_val2 = X_val[self.sens_attrs[0]]
        S_val = S_val2.to_numpy().reshape((S_val2.to_numpy().shape[0],))
        X_train = X_train.loc[:, X_train.columns != self.sens_attrs[0]].to_numpy()
        X_val = X_val.loc[:, X_val.columns != self.sens_attrs[0]].to_numpy()

        y_train = y_train2.to_numpy().reshape((y_train2.to_numpy().shape[0],))
        y_val = y_val2.to_numpy().reshape((y_val2.to_numpy().shape[0],))

        X_sample2, y_sample2, s_sample2 = model.fit(X_train, y_train, S_train, X_val, S_val)

        X_test = self.X_test.loc[:, self.X_test.columns != self.sens_attrs[0]].to_numpy()
        pred = model.predict(X_test)

        cols = self.X_train.columns.tolist()
        cols.remove(self.sens_attrs[0])
        idx = self.X_train.index.tolist()

        X_sample = pd.DataFrame(X_sample2, columns=cols)
        y_sample = pd.Series(y_sample2, name=self.label)
        s_sample = pd.Series(s_sample2, name=self.sens_attrs[0])
        
        X_sample = pd.concat([X_sample, s_sample], axis=1)


        if do_eval:
            metric_val = acc_bias(self.dataset_test, np.asarray(pred).reshape(-1,1), self.unprivileged_groups, self.privileged_groups, self.metric)
        else:
            metric_val = None

        return pred, X_sample, y_sample, metric_val, "FS", model


    def pfr(self, classifier, gamma, do_eval=False):
        """...
        """
        similarity_params = dict()
        similarity_matrix = "knn"

        similarity_params["num_hash"], similarity_params["num_table"], similarity_params["theta"] = 1, 10, 0.05
        similarity_params["k"], similarity_params["threshold"]  = 20, 3

        X_train, X_val, y_train, y_val = train_test_split(self.X_train, self.y_train, test_size=0.3, random_state=42)
        y_train = y_train.to_numpy().reshape((y_train.to_numpy().shape[0],))
        y_val = y_val.to_numpy().reshape((y_val.to_numpy().shape[0],))

        X_test = copy.deepcopy(self.X_test)

        if len(self.sens_attrs) > 1:
            gdf_train = X_train.groupby(self.sens_attrs)
            gdf_val = X_val.groupby(self.sens_attrs)
            gdf_test = X_test.groupby(self.sens_attrs)
            X_train["sensitive"] = pd.Series(dtype='int')
            X_test["sensitive"] = pd.Series(dtype='int')
            X_val["sensitive"] = pd.Series(dtype='int')
            c = 0
            for key, item in gdf_train:
                pdf_train = gdf_train.get_group(key)
                pdf_val = gdf_val.get_group(key)
                pdf_test = gdf_test.get_group(key)
                
                for i, row in pdf_train.iterrows():
                    X_train.loc[i, "sensitive"] = c
                for i, row in pdf_val.iterrows():
                    X_val.loc[i, "sensitive"] = c
                for i, row in pdf_test.iterrows():
                    X_test.loc[i, "sensitive"] = c
                c += 1

            for sens in self.sens_attrs:
                X_train.drop(sens, axis=1, inplace=True)
                X_test.drop(sens, axis=1, inplace=True)
                X_val.drop(sens, axis=1, inplace=True)

            sens_attrs = ["sensitive"]
            all_cols = X_train.columns.tolist()
            cols = all_cols + sens_attrs
        else:
            sens_attrs = copy.deepcopy(self.sens_attrs)

            all_cols = self.X_train.columns.tolist()
            for sens in sens_attrs:
                all_cols.remove(sens)
            cols = all_cols + sens_attrs

        for col in cols:
            X_train[col] = X_train[col].astype(float)
            X_test[col] = X_test[col].astype(float)
            X_val[col] = X_val[col].astype(float)

        if self.remove:
            for sens in sens_attrs:
                X_train = X_train.drop(sens, axis=1)
                X_test = X_test.drop(sens, axis=1)
                X_val = X_val.drop(sens, axis=1)
        X_train = X_train.to_numpy()
        X_test = X_test.to_numpy()
        X_val = X_val.to_numpy()

        w_train, edge_train, w_edge_train = generate_sim_matrix(X_train, similarity_matrix, similarity_params)
        w_test, edge_test, w_edge_test = generate_sim_matrix(X_test, similarity_matrix, similarity_params)
        w_val, edge_val, w_edge_val = generate_sim_matrix(X_val, similarity_matrix, similarity_params)

        model_type = "LogisticRegression"

        k_dim = estimate_dim(X_train)
        w_pfr = similarity_pfr(X_train, k_dim)

        #for gamma in [0, 0.001, 0.005, 0.01, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]:
        PFR_model = PFR(k = k_dim, W_s = w_pfr, W_F = w_train, gamma=gamma)
        PFR_model.fit(X_train)

        transf_x_train = PFR_model.transform(X_train)
        transf_x_test = PFR_model.transform(X_test)
        transf_x_val = PFR_model.transform(X_val)

        model = Model(model_type, transf_x_train, y_train, edge_train, w_edge_train, transf_x_test, self.y_te_np, edge_test, w_edge_test, transf_x_val, y_val)
        pred, train_performance, test_performance = model.train()

        if do_eval:
            metric_val = acc_bias(self.dataset_test, np.asarray(pred).reshape(-1,1), self.unprivileged_groups, self.privileged_groups, self.metric)
        else:
            metric_val = None

        return pred, transf_x_train, self.y_train, transf_x_test, metric_val, "PFR", PFR_model


    def iflipper(self, classifier, num_plot, do_eval=False):
        """...
        """
        similarity_params = dict()
        similarity_matrix = "knn"

        X_train, X_val, y_train, y_val = train_test_split(self.X_train, self.y_train, test_size=0.3, random_state=42)
        y_train = y_train.to_numpy().reshape((y_train.to_numpy().shape[0],))
        y_val = y_val.to_numpy().reshape((y_val.to_numpy().shape[0],))

        X_test = copy.deepcopy(self.X_test)

        X_train_return = copy.deepcopy(X_train)

        X_train = X_train.loc[:, X_train.columns != self.sens_attrs[0]]
        X_test = X_test.loc[:, X_test.columns != self.sens_attrs[0]]
        X_val = X_val.loc[:, X_val.columns != self.sens_attrs[0]]

        cols = self.X_train.columns.tolist()
        for sens in self.sens_attrs:
            cols.remove(sens)
        for col in cols:
            X_train[col] = X_train[col].astype(float)
            X_test[col] = X_test[col].astype(float)
            X_val[col] = X_val[col].astype(float)
        X_train = X_train.to_numpy()
        X_test = X_test.to_numpy()
        X_val = X_val.to_numpy()

        similarity_params["num_hash"], similarity_params["num_table"], similarity_params["theta"] = 1, 10, 0.05
        similarity_params["k"], similarity_params["threshold"]  = 20, 3

        w_train, edge_train, w_edge_train = generate_sim_matrix(X_train, similarity_matrix, similarity_params)
        w_test, edge_test, w_edge_test = generate_sim_matrix(X_test, similarity_matrix, similarity_params)
        w_val, edge_val, w_edge_val = generate_sim_matrix(X_val, similarity_matrix, similarity_params)

        model_type = "LogisticRegression"

        init_error = measure_error(y_train, edge_train, w_edge_train)

        m = (init_error * num_plot)
        
        IFLIP = iFlipper(y_train, w_train, edge_train, w_edge_train)
        flipped_label = IFLIP.transform(m)

        model = Model(model_type, X_train, flipped_label, edge_train, w_edge_train, X_test, self.y_te_np, edge_test, w_edge_test, X_val, y_val)
        pred, train_performance, test_performance = model.train()

        y_train = pd.Series(flipped_label, name=self.label)

        if do_eval:
            metric_val = acc_bias(self.dataset_test, np.asarray(pred).reshape(-1,1), self.unprivileged_groups, self.privileged_groups, self.metric)
        else:
            metric_val = None

        return pred, X_train_return, y_train, metric_val, "iFlipper", IFLIP

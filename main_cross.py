"""
main method
"""
import os
import warnings
import argparse
import shelve
import ast
import copy
import math
import itertools
import json
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from aif360.datasets import BinaryLabelDataset
from aif360.metrics import BinaryLabelDatasetMetric
import algorithm

warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.filterwarnings('ignore')

parser = argparse.ArgumentParser()
parser.add_argument("--ds", type=str, help="Name of the input .csv file.")
parser.add_argument("-o", "--output", type=str, help="Directory of the generated output files.")
parser.add_argument("--testsize", default=0.5, type=float, help="Dataset is randomly split into\
    training and test datasets. This value indicates the size of the test dataset. Default value: 0.5")
parser.add_argument("--index", default="index", type=str, help="Column name containing the index\
    of each entry. Default given column name: index.")
parser.add_argument("--sensitive", type=str, help="List of column names of the sensitive attributes.")
parser.add_argument("--favored", type=str, help="Tuple of values of privileged group.")
parser.add_argument("--label", type=str, help="Column name of the target value.")
parser.add_argument("--metric", default="mean", type=str, help="Metric which will be used to test\
    the classifier combinations. Default metric: mean.")
parser.add_argument("--randomstate", default=-1, type=int, help="Randomstate of the splits.")
parser.add_argument("--pre_models", default=None, type=str, help="List of models that should be trained.")
parser.add_argument("--in_models", default=None, type=str, help="List of models that should be trained.")
parser.add_argument("--post_models", default=None, type=str, help="List of models that should be trained.")
parser.add_argument("--tuning", default="False", type=str, help="Set to True if hyperparameter\
    tuning should be performed. Else, default parameter values are used. Default: False")
args = parser.parse_args()

input_file = args.ds
link = args.output
testsize = float(args.testsize)
index = args.index
sens_attrs = ast.literal_eval(args.sensitive)
favored = ast.literal_eval(args.favored)
label = args.label
metric = args.metric
randomstate = args.randomstate
if randomstate == -1:
    import random
    randomstate = random.randint(1,1000)
pre_model_list = ast.literal_eval(args.pre_models)
in_model_list = ast.literal_eval(args.in_models)
post_model_list = ast.literal_eval(args.post_models)
tuning = args.tuning == "True"
if tuning:
    do_eval = True
else:
    do_eval = False
combine = True

df = pd.read_csv("Datasets/" + input_file + ".csv", index_col=index)
grouped_df = df.groupby(sens_attrs)
group_keys = grouped_df.groups.keys()

privileged_groups = []
unprivileged_groups = []
priv_dict = dict()
unpriv_dict = dict()

if isinstance(favored, tuple):
    for i, fav_val in enumerate(favored):
        priv_dict[sens_attrs[i]] = fav_val
        all_val = list(df.groupby(sens_attrs[i]).groups.keys())
        for poss_val in all_val:
            if poss_val != fav_val:
                unpriv_dict[sens_attrs[i]] = poss_val
else:
    if favored == 0:
        priv_dict[sens_attrs[0]] = 0
        unpriv_dict[sens_attrs[0]] = 1
    elif favored == 1:
        priv_dict[sens_attrs[0]] = 1
        unpriv_dict[sens_attrs[0]] = 0

privileged_groups = [priv_dict]
unprivileged_groups = [unpriv_dict]

#Read the input dataset & split it into training, test & prediction dataset.
#Prediction dataset only needed for evaluation, otherwise size is automatically 0.
X = df.loc[:, df.columns != label]
y = df[label]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=testsize,
    random_state=randomstate)

train_df = pd.merge(X_train, y_train, left_index=True, right_index=True)
test_df = pd.merge(X_test, y_test, left_index=True, right_index=True)
dataset_train = BinaryLabelDataset(df=train_df, label_names=[label], protected_attribute_names=sens_attrs)
dataset_test = BinaryLabelDataset(df=test_df, label_names=[label], protected_attribute_names=sens_attrs)
full_dataset = BinaryLabelDataset(df=df, label_names=[label], protected_attribute_names=sens_attrs)

train_id_list = []
for i, row in X_train.iterrows():
    train_id_list.append(i)

test_id_list = []
for i, row in X_test.iterrows():
    test_id_list.append(i)

y_train = y_train.to_frame()
y_test = y_test.to_frame()
result_df = copy.deepcopy(y_test)
for sens in sens_attrs:
    result_df[sens] = X_test[sens]

preproc_init = algorithm.Preprocessing(X_train, X_test, y_train, y_test, sens_attrs, favored, 
    dataset_train, dataset_test, testsize, randomstate, privileged_groups, unprivileged_groups,
    index, label, metric, combine, link, input_file, False)
aif_preproc_init = algorithm.AIFPreprocessing(full_dataset, dataset_train, dataset_test, X_test,
    testsize, randomstate, sens_attrs, privileged_groups, unprivileged_groups, label, metric, link, False)
inproc_init = algorithm.Inprocessing(X_train, X_test, y_train, y_test, sens_attrs, favored,
    dataset_train, dataset_test, privileged_groups, unprivileged_groups, label, metric, link, False)
aif_inproc_init = algorithm.AIFInprocessing(dataset_train, dataset_test, sens_attrs,
    privileged_groups, unprivileged_groups, label, metric, link, False)
postproc_init = algorithm.Postprocessing(X_train, X_test, y_train, y_test, sens_attrs, favored,
    dataset_train, dataset_test, privileged_groups, unprivileged_groups, label, metric, link, False)
aif_postproc_init = algorithm.AIFPostprocessing(X_train, X_test, y_train, y_test, sens_attrs,
    dataset_train, dataset_test, privileged_groups, unprivileged_groups, label, metric,
    link, False)

log_regr = LogisticRegression(solver='lbfgs')
log_regr2 = LogisticRegression(C=1.0, penalty="l2", solver="liblinear", max_iter=100)
dectree = DecisionTreeClassifier()

params = json.load(open('configs/params.json'))
inpost_model_list = in_model_list + post_model_list
prein_model_list = pre_model_list + in_model_list

for pre in pre_model_list:
    print(pre)
    preproc = copy.deepcopy(preproc_init)
    aif_preproc = copy.deepcopy(aif_preproc_init)
    if tuning:
        paramlist = list(params[pre]["tuning"].keys())
        parameters = []
        for param in paramlist:
            parameters.append(params[pre]["tuning"][param])
        full_list = list(itertools.product(*parameters))
        do_eval = True
    else:
        paramlist = list(params[pre]["default"].keys())
        li = []
        for param in paramlist:
            li.append(params[pre]["default"][param]) 
        full_list = [li]
        do_eval = False
    real = [item for sublist in dataset_test.labels.tolist() for item in sublist]
    max_val = 0
    best_li = 0

    for i, li in enumerate(full_list):
        score = 0
        #func = eval(params[model]["method"])
        try:
            if "LFR" == pre:
                prediction, X_train, y_train, X_test_adapt, score, pre_model, clf = aif_preproc.lfr(log_regr, k=li[2], Ay=li[0], Az=li[1], do_eval=do_eval)
            elif "Fair-SMOTE" == pre:
                prediction, X_train, y_train, score, pre_model, clf = preproc.smote(log_regr2, cr=li[0], f=li[1], do_eval=do_eval)
            elif "FairSSL-Lx" == pre:
                prediction, X_train, y_train, score, pre_model, clf = preproc.fairssl(log_regr2, ssl_type=li[3], balancing=li[0]=="True", cr=li[1], f=li[2], do_eval=do_eval)
            elif "FairSSL-xT" == pre:
                prediction, X_train, y_train, score, pre_model, clf = preproc.fairssl(log_regr2, ssl_type=li[3], balancing=li[0]=="True", cr=li[1], f=li[2], do_eval=do_eval)
            elif "LTDD" == pre:
                prediction, X_train, y_train, X_test_adapt, score, pre_model, clf = preproc.ltdd(log_regr2, do_eval=do_eval)
            elif "FairRR" == pre:
                prediction, X_train, y_train, score, pre_model, clf = preproc.fair_rr(log_regr, level=li[0], do_eval=do_eval)
            elif "FS" == pre:
                prediction, X_train, y_train, score, pre_model, clf = preproc.fesf(K=li[0], do_eval=do_eval)
            elif "iFlipper" == pre:
                prediction, X_train, y_train, score, pre_model, clf = preproc.iflipper(log_regr, num_plot=li[0], do_eval=do_eval)
        except Exception as e:
            print("------------------")
            print(pre)
            print(e)
            continue

        if (not tuning) or (score > max_val):
            max_val = score
            best_li = li
            best_pred = prediction
            X_train2 = copy.deepcopy(X_train)
            if isinstance(y_train, pd.DataFrame):
                y_train2 = copy.deepcopy(y_train)
            else:
                y_train2 = copy.deepcopy(y_train.to_frame())

    try:
        new_df = pd.concat([X_train2, y_train2],ignore_index=False,axis=1)
    except:
        X_train2 = X_train2.reset_index(drop=True)
        y_train2 = y_train2.reset_index(drop=True)
        new_df = pd.concat([X_train2, y_train2],ignore_index=False,axis=1)
        
    df_link = "Datasets/" + input_file + "/" + metric + "/"
    try:
        os.makedirs(df_link)
    except FileExistsError:
        pass
    new_df.to_csv(df_link + pre + "_" + str(tuning) + ".csv", index_label="index")

    if (not tuning) or (max_val > 0):
        for inpost in inpost_model_list:
            print(pre + "_" + inpost)
            if pre in ("LFR", "LTDD", "PFR"):
                X_test2 = copy.deepcopy(X_test_adapt)
            else:
                X_test2 = copy.deepcopy(X_test)
            train_df = pd.merge(X_train2, y_train2, left_index=True, right_index=True)
            dataset_train = BinaryLabelDataset(df=train_df, label_names=[label], protected_attribute_names=sens_attrs)
            test_df = pd.merge(X_test2, y_test, left_index=True, right_index=True)
            if test_df.shape[0] == 0:
                X_test2 = X_test2.reset_index(drop=True)
                y_test = y_test.reset_index(drop=True)
                test_df = pd.merge(X_test2, y_test, left_index=True, right_index=True)
            dataset_test = BinaryLabelDataset(df=test_df, label_names=[label], protected_attribute_names=sens_attrs)
            preproc2 = algorithm.Preprocessing(X_train2, X_test2, y_train2, y_test, sens_attrs, favored, 
                dataset_train, dataset_test, testsize, randomstate, privileged_groups, unprivileged_groups,
                index, label, metric, combine, link, input_file, False)
            inproc = algorithm.Inprocessing(X_train2, X_test2, y_train2, y_test, sens_attrs, favored,
                dataset_train, dataset_test, privileged_groups, unprivileged_groups, label, metric, link, False)
            postproc = algorithm.Postprocessing(X_train2, X_test2, y_train2, y_test, sens_attrs, favored,
                dataset_train, dataset_test, privileged_groups, unprivileged_groups, label, metric, link, False)
            aif_preproc2 = algorithm.AIFPreprocessing(full_dataset, dataset_train, dataset_test, X_test2,
                testsize, randomstate, sens_attrs, privileged_groups, unprivileged_groups, label, metric, link, False)
            aif_inproc = algorithm.AIFInprocessing(dataset_train, dataset_test, sens_attrs,
                privileged_groups, unprivileged_groups, label, metric, link, False)
            aif_postproc = algorithm.AIFPostprocessing(X_train2, X_test2, y_train2, y_test, sens_attrs,
                dataset_train, dataset_test, privileged_groups, unprivileged_groups, label, metric,
                link, False)

            paramlist = list(params[inpost]["default"].keys())
            li = []
            if not tuning:
                for param in paramlist:
                    li.append(params[inpost]["default"][param])
                    full_list = [li]
            else:
                for param in paramlist:
                    li.append(params[inpost]["tuning"][param])
                full_list = list(itertools.product(*li))

            max_val = 0
            best_li = 0
            for i, li in enumerate(full_list):
                score = 0
                #func = eval(params[model]["method"])
                try:
                    if "AdversarialDebiasing" == inpost:
                        prediction, score, model, clf = aif_inproc.adversarial_debiasing("plain_classifier", weight=li[0], debias=li[1]=="True", do_eval=do_eval)
                    elif "GerryFairClassifier" == inpost:
                        prediction, score, model, clf = aif_inproc.gerryfair(log_regr, gamma=li[0], do_eval=do_eval)
                    elif "MetaFairClassifier" == inpost:
                        prediction, score, model, clf = aif_inproc.metafair(tau=li[0], do_eval=do_eval)
                    elif "PrejudiceRemover" == inpost:
                        prediction, score, model, clf = aif_inproc.prejudice_remover(eta=li[0], do_eval=do_eval)
                    elif "ExponentiatedGradientReduction" == inpost:
                        prediction, score, model, clf = aif_inproc.exponentiated_gradient_reduction(log_regr, eps=li[0], eta=li[1], drop_prot_attr=li[2]=="True", do_eval=do_eval)
                    elif "GridSearchReduction" == inpost:
                        prediction, score, model, clf = aif_inproc.gridsearch_reduction(log_regr, weight=li[0], drop_prot_attr=li[1]=="True", do_eval=do_eval)
                    elif "EqOddsPostprocessing" == inpost:
                        prediction, score, model, clf = aif_postproc.eqodds_postproc(log_regr, do_eval=do_eval)
                    elif "CalibratedEqOddsPostprocessing" == inpost:
                        prediction, score, model, clf = aif_postproc.calibrated_eqodds_postproc(log_regr, do_eval=do_eval)
                    elif "RejectOptionClassification" == inpost:
                        prediction, score, model, clf = aif_postproc.reject_option_class(log_regr, eps=li[0], do_eval=do_eval)
                    elif "FaX" == inpost:
                        prediction, score, model, clf = postproc.fax(method="MIM", do_eval=do_eval)
                    elif "DPAbstention" == inpost:
                        prediction, score, model, clf = postproc.dpabst(log_regr, alpha=li[0], do_eval=do_eval)
                    elif "GetFair" == inpost:
                        prediction, score, model, clf = postproc.getfair(log_regr, lam=li[0], step_size=li[1], episodes=li[2], hidden_size=li[3], layers=li[4], do_eval=do_eval)
                    elif "FairBayesDPP" == inpost:
                        prediction, score, model, clf = postproc.fair_bayes_dpp(n_epochs=li[0], lr=li[1], batch_size=li[2], n_seeds=li[3], do_eval=do_eval)
                    elif "JiangNachum" == inpost:
                        prediction, X_train3, y_train3, weights, score, model, clf = postproc.jiang_nachum(log_regr, iterations=li[0], learning=li[1], do_eval=do_eval)
                    elif "AdaFair" == inpost:
                        prediction, score, model, clf = inproc.adafair(dectree, iterations=li[0], learning_rate=li[1], do_eval=do_eval)
                    elif "FairGeneralizedLinearModel" == inpost:
                        prediction, score, model, clf = inproc.fglm(lam=li[0], family=li[1], discretization=li[2], do_eval=do_eval)
                    elif "SquaredDifferenceFairLogistic" == inpost:
                        prediction, score, model, clf = inproc.squared_diff_fair_logistic(lam=li[0], do_eval=do_eval)
                    elif "FairnessConstraintModel" == inpost:
                        prediction, score, model, clf = inproc.fairness_constraint_model(c=li[0], tau=li[1], mu=li[2], eps=li[3], do_eval=do_eval)
                    elif "DisparateMistreatmentModel" == inpost:
                        prediction, score, model, clf = inproc.disparate_treatment_model(c=li[0], tau=li[1], mu=li[2], eps=li[3], do_eval=do_eval)
                    elif "ConvexFrameworkModel" == inpost:
                        prediction, score, model, clf = inproc.convex_framework(lam=li[0], family=li[1], penalty=li[2], do_eval=do_eval)
                    elif "HSICLinearRegression" == inpost:
                        prediction, score, model, clf = inproc.hsic_linear_regression(lam=li[0], do_eval=do_eval)
                    elif "GeneralFairERM" == inpost:
                        prediction, score, model, clf = inproc.general_ferm(eps=li[0], k=li[1], do_eval=do_eval)
                    elif "FAGTB" == inpost:
                        prediction, score, model, clf = inproc.fagtb(estimators=li[0], learning=li[1], lam=li[2], do_eval=do_eval)
                    elif "FairDummies" == inpost:
                        prediction, score, model, clf = inproc.fair_dummies(batch=li[0], lr=li[1], mu=li[2], second_scale=li[3], epochs=li[4], model_type=li[5], do_eval=do_eval)
                    elif "HGR" == inpost:
                        prediction, score, model, clf = inproc.hgr(batch=li[0], lr=li[1], mu=li[2], epochs=li[3], model_type=li[4], do_eval=do_eval)
                    elif "MultiAdversarialDebiasing" == inpost:
                        prediction, score, model, clf = inproc.multi_adv_deb(weight=li[0], do_eval=do_eval)
                    elif "GradualCompatibility" == inpost:
                        prediction, score, model, clf = inproc.grad_compat(reg=li[0], reg_val=li[1], weights_init=li[2], lambda_=li[3], do_eval=do_eval)
                    elif "LevEqOpp" == inpost:
                        prediction, score, model, clf = postproc.lev_eq_opp(log_regr, do_eval=do_eval)
                    elif "GroupDebias" == inpost:
                        prediction, score, model, clf = postproc.group_debias(log_regr, method=li[0], n_estimators=li[1], uniformity=li[2], bootstrap=li[3]=="True", do_eval=do_eval)
                    elif "fairret" == inpost:
                        prediction, score, model, clf = inproc.fairret(lam=li[0], lr=li[1], h_layer=li[2], do_eval=do_eval)

                    if not tuning:
                        result_df[pre + "_" + model] = prediction
                        result_df.to_csv(link + pre + "_" + model + "_prediction.csv")
                        result_df = result_df.drop(columns=[pre + "_" + model])

                    else:
                        if score > max_val:
                            max_val = copy.deepcopy(score)
                            best_li = li
                            #best_pred = prediction

                            result_df[pre + "_" + model + "_tuned"] = prediction
                            result_df.to_csv(link + pre + "_" + model + "_tuned_prediction.csv")
                            result_df = result_df.drop(columns=[pre + "_" + model + "_tuned"])

                except Exception as e:
                    print("------------------")
                    print(pre + "_" + inpost)
                    print(e)
                    print("------------------")

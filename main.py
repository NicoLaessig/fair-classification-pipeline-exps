"""
main method
"""
import warnings
import argparse
import ast
import copy
import math
import itertools
import json
import subprocess
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from aif360.datasets import BinaryLabelDataset
from aif360.metrics import BinaryLabelDatasetMetric
from imblearn.combine import SMOTETomek
from sklearn.feature_selection import VarianceThreshold, RFECV
from sklearn.decomposition import PCA
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
parser.add_argument("--models", default=None, type=str, help="List of models that should be trained.")
parser.add_argument("--tuning", default="False", type=str, help="Set to True if hyperparameter\
    tuning should be performed. Else, default parameter values are used. Default: False")
parser.add_argument("--removal", default="False", type=str, help="Set to True if protected attributes\
    should be removed.")
parser.add_argument("--binarize", default="False", type=str, help="Set to True if protected attributes\
    should be binarized.")
parser.add_argument("--balance", default="False", type=str, help="If balance should be applied.")
parser.add_argument("--fselection", default="False", type=str, help="Set to True if feature selection activated.")
parser.add_argument("--dimred", default="False", type=str, help="Set to True if dimensionality reduction is activated.")
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
model_list = ast.literal_eval(args.models)
tuning = args.tuning == "True"
rem_prot = args.removal == "True"
binarize = args.binarize == "True"
balance = args.balance
fselection = args.fselection
dimred = args.dimred == "True"
combine = False

df = pd.read_csv("Datasets/" + input_file + ".csv", index_col=index)
grouped_df = df.groupby(sens_attrs)
group_keys = grouped_df.groups.keys()

##DATA PREPARATION SETTINGS START
if binarize:
    if len(sens_attrs) > 1:
        for i, row in df.iterrows():
            fav = True
            for j, sens in enumerate(sens_attrs):
                if row[sens] != favored[j]:
                    fav = False
                    df.loc[i, "sensitive"] = 0
                    break
            if fav:
                df.loc[i, "sensitive"] = 1
        for sens in sens_attrs:
            df = df.drop(sens, axis=1)
        sens_attrs = ["sensitive"]
        favored = (1)

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

if balance == "classic":
    smttmk = SMOTETomek()
    X_train, y_train = smttmk.fit_resample(X_train, y_train)
elif balance == "protected":
    smttmk = SMOTETomek()
    df_new = X_train.merge(y_train, left_index=True, right_index=True, how="inner")
    key_list = []
    cols = copy.deepcopy(sens_attrs)
    cols.append(label)
    grouped_df = df_new.groupby(cols)
    count = 0
    for key, item in grouped_df:
        part_df = grouped_df.get_group(key)
        for i, row in part_df.iterrows():
            df_new.loc[i, "target"] = count
        count += 1
        key_list.append(key)

    X_train = df_new.loc[:, df_new.columns != "target"]
    y_train = df_new["target"]
    for col in cols:
        X_train = X_train.drop(col, axis=1)

    X_train, y_train = smttmk.fit_resample(X_train, y_train)
    df_new = X_train.merge(y_train, left_index=True, right_index=True, how="inner")

    grouped_df = df_new.groupby("target")
    for i in range(count):
        part_df = grouped_df.get_group(i)
        for r, row in part_df.iterrows():
            for j, col in enumerate(cols):
                df_new.loc[r, col] = key_list[i][j]

    df_new = df_new.drop("target", axis=1)

    X_train = df_new.loc[:, df_new.columns != label]
    y_train = df_new[label]

    cols = X_test.columns.to_list()
    X_train = X_train[cols]


if dimred:
    prot_feats_train = []
    prot_feats_test = []
    X_train_no_prot = copy.deepcopy(X_train)
    X_test_no_prot = copy.deepcopy(X_test)
    for sens in sens_attrs:
        prot_feats_train.append(X_train[sens])
        prot_feats_test.append(X_test[sens])
        X_train_no_prot = X_train_no_prot.loc[:, X_train_no_prot.columns != sens]
        X_test_no_prot = X_test_no_prot.loc[:, X_test_no_prot.columns != sens]
    pca = PCA(n_components="mle")
    pca.fit(X_train_no_prot)
    X_train_np = pca.transform(X_train_no_prot)
    X_test_np = pca.transform(X_test_no_prot)
    cols = [str(i) for i in range(len(X_train_np[0]))]
    X_train = pd.DataFrame(X_train_np, columns=cols)
    X_test = pd.DataFrame(X_test_np, columns=cols)

    for i, sens in enumerate(sens_attrs):
        X_train[sens] = prot_feats_train[i].tolist()
        X_test[sens] = prot_feats_test[i].tolist()

    X_train["index"] = prot_feats_train[0].index.values.tolist()
    X_test["index"] = prot_feats_test[0].index.values.tolist()
    X_train.set_index("index", inplace=True)
    X_test.set_index("index", inplace=True)


if fselection == "VT":
    cols = list(X_train.columns)
    selector = VarianceThreshold()
    selector.fit(X_train)
    variances = selector.variances_
    rm_list = []
    for i, col in enumerate(cols):
        if col not in sens_attrs and variances[i] < 0.1:
            rm_list.append(col)
    if len(rm_list) >= 1:
        for rm in rm_list:
            X_train = X_train.loc[:, X_train.columns != rm]
            X_test = X_test.loc[:, X_test.columns != rm]
elif fselection == "RFECV":
    cols = list(X_train.columns)
    rfe = RFECV(LogisticRegression(), cv=10)
    rfe.fit(X_train, y_train)
    support = rfe.support_
    rm_list = []
    for i, col in enumerate(cols):
        if col not in sens_attrs and not support[i]:
            rm_list.append(col)
    if len(rm_list) >= 1:
        for rm in rm_list:
            X_train = X_train.loc[:, X_train.columns != rm]
            X_test = X_test.loc[:, X_test.columns != rm]

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
##DATA PREPARATION SETTINGS END

preproc = algorithm.Preprocessing(X_train, X_test, y_train, y_test, sens_attrs, favored, 
    dataset_train, dataset_test, testsize, randomstate, privileged_groups, unprivileged_groups,
    index, label, metric, combine, link, input_file, rem_prot)
inproc = algorithm.Inprocessing(X_train, X_test, y_train, y_test, sens_attrs, favored,
    dataset_train, dataset_test, privileged_groups, unprivileged_groups, label, metric, link, rem_prot)
postproc = algorithm.Postprocessing(X_train, X_test, y_train, y_test, sens_attrs, favored,
    dataset_train, dataset_test, privileged_groups, unprivileged_groups, label, metric, link, rem_prot)
aif_preproc = algorithm.AIFPreprocessing(full_dataset, dataset_train, dataset_test, X_test,
    testsize, randomstate, sens_attrs, privileged_groups, unprivileged_groups, label, metric, link, rem_prot)
aif_inproc = algorithm.AIFInprocessing(dataset_train, dataset_test, sens_attrs,
    privileged_groups, unprivileged_groups, label, metric, link, rem_prot)
aif_postproc = algorithm.AIFPostprocessing(X_train, X_test, y_train, y_test, sens_attrs,
    dataset_train, dataset_test, privileged_groups, unprivileged_groups, label, metric,
    link, rem_prot)

log_regr = LogisticRegression(solver='lbfgs')
log_regr2 = LogisticRegression(C=1.0, penalty="l2", solver="liblinear", max_iter=100)
dectree = DecisionTreeClassifier()

params = json.load(open('configs/params.json'))

for model in model_list:
    if tuning:
        paramlist = list(params[model]["tuning"].keys())
        parameters = []
        for param in paramlist:
            parameters.append(params[model]["tuning"][param])
        full_list = list(itertools.product(*parameters))
        do_eval = True
    else:
        paramlist = list(params[model]["default"].keys())
        li = []
        for param in paramlist:
            li.append(params[model]["default"][param]) 
        full_list = [li]
        do_eval = False
    
    real = [item for sublist in dataset_test.labels.tolist() for item in sublist]
    max_val = 0
    best_li = 0

    for i, li in enumerate(full_list):
        score = 0
        #func = eval(params[model]["method"])
        try:
            #AIF Preprocessing Predictions
            if "DisparateImpactRemover" == model:
                prediction, X_train2, y_train2, X_test2, score, model, clf = aif_preproc.disparate_impact_remover(log_regr, repair=li[0], do_eval=do_eval)
            elif "LFR" == model:
                prediction, X_train2, y_train2, X_test_adapt, score, model, clf = aif_preproc.lfr(log_regr, k=li[2], Ay=li[0], Az=li[1], do_eval=do_eval)
            elif "Reweighing" == model:
                prediction, X_train2, y_train2, sample_weight, score, model, clf = aif_preproc.reweighing(log_regr, do_eval=do_eval)
            #AIF Inprocessing Predictions
            elif "AdversarialDebiasing" == model:
                prediction, score, model, clf = aif_inproc.adversarial_debiasing("plain_classifier", weight=li[0], debias=li[1]=="True", do_eval=do_eval)
            elif "GerryFairClassifier" == model:
                prediction, score, model, clf = aif_inproc.gerryfair(log_regr, gamma=li[0], do_eval=do_eval)
            elif "MetaFairClassifier" == model:
                prediction, score, model, clf = aif_inproc.metafair(tau=li[0], do_eval=do_eval)
            elif "PrejudiceRemover" == model:
                prediction, score, model, clf = aif_inproc.prejudice_remover(eta=li[0], do_eval=do_eval)
            elif "ExponentiatedGradientReduction" == model:
                prediction, score, model, clf = aif_inproc.exponentiated_gradient_reduction(log_regr, eps=li[0], eta=li[1], drop_prot_attr=li[2]=="True", do_eval=do_eval)
            elif "GridSearchReduction" == model:
                prediction, score, model, clf = aif_inproc.gridsearch_reduction(log_regr, weight=li[0], drop_prot_attr=li[1]=="True", do_eval=do_eval)
            #AIF Postprocessing Predictions
            elif "EqOddsPostprocessing" == model:
                prediction, score, model, clf = aif_postproc.eqodds_postproc(log_regr, do_eval=do_eval)
            elif "CalibratedEqOddsPostprocessing" == model:
                prediction, score, model, clf = aif_postproc.calibrated_eqodds_postproc(log_regr, do_eval=do_eval)
            elif "RejectOptionClassification" == model:
                prediction, score, model, clf = aif_postproc.reject_option_class(log_regr, eps=li[0], do_eval=do_eval)
            #Other Preprocessing Predictions
            elif "Fair-SMOTE" == model:
                prediction, X_train2, y_train2, score, model, clf = preproc.smote(log_regr2, cr=li[0], f=li[1], do_eval=do_eval)
            elif "FairSSL-Lx" == model:
                prediction, X_train2, y_train2, score, wrong_modelname, clf = preproc.fairssl(log_regr2, ssl_type=li[3], balancing=li[0]=="True", cr=li[1], f=li[2], do_eval=do_eval)
            elif "FairSSL-xT" == model:
                prediction, X_train2, y_train2, score, wrong_modelname, clf = preproc.fairssl(log_regr2, ssl_type=li[3], balancing=li[0]=="True", cr=li[1], f=li[2], do_eval=do_eval)
            elif "LTDD" == model:
                prediction, X_train2, y_train2, X_test_adapt, score, model, clf = preproc.ltdd(log_regr2, do_eval=do_eval)
            #Other Inprocessing predictions
            elif "AdaFair" == model:
                prediction, score, model, clf = inproc.adafair(dectree, iterations=li[0], learning_rate=li[1], do_eval=do_eval)
            elif "FairGeneralizedLinearModel" == model:
                prediction, score, model, clf = inproc.fglm(lam=li[0], family=li[1], discretization=li[2], do_eval=do_eval)
            elif "SquaredDifferenceFairLogistic" == model:
                prediction, score, model, clf = inproc.squared_diff_fair_logistic(lam=li[0], do_eval=do_eval)
            elif "FairnessConstraintModel" == model:
                prediction, score, model, clf = inproc.fairness_constraint_model(c=li[0], tau=li[1], mu=li[2], eps=li[3], do_eval=do_eval)
            elif "DisparateMistreatmentModel" == model:
                prediction, score, model, clf = inproc.disparate_treatment_model(c=li[0], tau=li[1], mu=li[2], eps=li[3], do_eval=do_eval)
            elif "ConvexFrameworkModel" == model:
                prediction, score, model, clf = inproc.convex_framework(lam=li[0], family=li[1], penalty=li[2], do_eval=do_eval)
            elif "HSICLinearRegression" == model:
                prediction, score, model, clf = inproc.hsic_linear_regression(lam=li[0], do_eval=do_eval)
            elif "GeneralFairERM" == model:
                prediction, score, model, clf = inproc.general_ferm(eps=li[0], k=li[1], do_eval=do_eval)
            elif "FAGTB" == model:
                prediction, score, model, clf = inproc.fagtb(estimators=li[0], learning=li[1], lam=li[2], do_eval=do_eval)
            elif "FairDummies" == model:
                prediction, score, model, clf = inproc.fair_dummies(batch=li[0], lr=li[1], mu=li[2], second_scale=li[3], epochs=li[4], model_type=li[5], do_eval=do_eval)
            elif "HGR" == model:
                prediction, score, model, clf = inproc.hgr(batch=li[0], lr=li[1], mu=li[2], epochs=li[3], model_type=li[4], do_eval=do_eval)  
            elif "MultiAdversarialDebiasing" == model:
                prediction, score, model, clf = inproc.multi_adv_deb(weight=li[0], do_eval=do_eval)
            elif "GradualCompatibility" == model:
                prediction, score, model, clf = inproc.grad_compat(reg=li[0], reg_val=li[1], weights_init=li[2], lambda_=li[3], do_eval=do_eval)
            #Other Postprocessing predictions
            elif "JiangNachum" == model:
                prediction, X_train2, y_train2, weights, score, model, clf = postproc.jiang_nachum(log_regr, iterations=li[0], learning=li[1], do_eval=do_eval)
            elif "FaX" == model:
                prediction, score, model, clf = postproc.fax(method="MIM", do_eval=do_eval)
            elif "DPAbstention" == model:
                prediction, score, model, clf = postproc.dpabst(log_regr, alpha=li[0], do_eval=do_eval)
            elif "GetFair" == model:
                prediction, score, model, clf = postproc.getfair(log_regr, lam=li[0], step_size=li[1], episodes=li[2], hidden_size=li[3], layers=li[4], do_eval=do_eval)
            elif "FairBayesDPP" == model:
                prediction, score, model, clf = postproc.fair_bayes_dpp(n_epochs=li[0], lr=li[1], batch_size=li[2], n_seeds=li[3], do_eval=do_eval)
            elif "LevEqOpp" == model:
                prediction, score, model, clf = postproc.lev_eq_opp(log_regr, do_eval=do_eval)
            elif "FairRR" == model:
                prediction, X_train2, y_train2, score, model, clf = preproc.fair_rr(log_regr, level=li[0], do_eval=do_eval)
            elif "GroupDebias" == model:
                prediction, score, model, clf = postproc.group_debias(log_regr, method=li[0], n_estimators=li[1], uniformity=li[2], bootstrap=li[3]=="True", do_eval=do_eval)
            elif "FS" == model:
                prediction, X_train2, y_train2, score, model, clf = preproc.fesf(K=li[0], do_eval=do_eval)
            elif "iFlipper" == model:
                prediction, X_train2, y_train2, score, model, clf = preproc.iflipper(log_regr, num_plot=li[0], do_eval=do_eval)
            elif "PFR" == model:
                prediction, X_train2, y_train2, X_test2, score, model, clf = preproc.pfr(log_regr, gamma=li[0], do_eval=do_eval)
            elif "fairret" == model:
                prediction, score, model, clf = inproc.fairret(lam=li[0], lr=li[1], h_layer=li[2], do_eval=do_eval)
            #LogRegr Baseline
            elif "LogisticRegression" in model_list and not tuning:
                lr_clf = copy.deepcopy(log_regr)
                lr_clf.fit(X_train, y_train)
                prediction = lr_clf.predict(X_test)
            elif "LogisticRegressionRemoved" in model_list and not tuning:
                X_train_rem = copy.deepcopy(X_train)
                X_test_rem = copy.deepcopy(X_test)
                for sens in sens_attrs:
                    X_train_rem = X_train_rem.loc[:, X_train_rem.columns != sens]
                    X_test_rem = X_test_rem.loc[:, X_test_rem.columns != sens]
                lr_clf = copy.deepcopy(log_regr)
                lr_clf.fit(X_train_rem, y_train)
                prediction = lr_clf.predict(X_test_rem)

        except Exception as e:
            print("------------------")
            prediction = None
            print(model)
            print(e)
            print("------------------")


        if tuning:
            if score > max_val:
                max_val = score
                best_li = li
                best_pred = prediction

            if prediction is not None:
                result_df[model + "_" + str(i)] = prediction
                result_df.to_csv(link + model + "_" + str(i) + "_prediction.csv")
                result_df = result_df.drop(columns=[model + "_" + str(i)])

    #Else no scoring returned
    if tuning and max_val > 0:
        result_df[model + "_tuned"] = best_pred
        result_df.to_csv(link + model + "_tuned_prediction.csv")
        result_df = result_df.drop(columns=[model + "_tuned"])
        
    elif prediction is not None:
        result_df[model] = prediction
        result_df.to_csv(link + model + "_prediction.csv")
        result_df = result_df.drop(columns=[model])

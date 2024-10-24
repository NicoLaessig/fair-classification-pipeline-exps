"""
This code evaluates the results of the experiments based on several metrics.
"""
import warnings
import argparse
import ast
import copy
import shelve
import pandas as pd
import time
import math
import numpy as np
from sklearn.neighbors import NearestNeighbors
from aif360.datasets import StandardDataset
from aif360.metrics import BinaryLabelDatasetMetric, ClassificationMetric

warnings.simplefilter(action='ignore', category=FutureWarning)

parser = argparse.ArgumentParser()
parser.add_argument("--ds", type=str, help="Name of the input .csv file.")
parser.add_argument("--folder", type=str, help="Directory of the generated output files.")
parser.add_argument("--index", default="index", type=str, help="Column name containing the index\
    of each entry. Default given column name: index.")
parser.add_argument("--sensitive", type=str, help="List of column names of the sensitive attributes.")
parser.add_argument("--label", type=str, help="Column name of the target value.")
parser.add_argument("--favored", default=None, type=str, help="Tuple of favored group.\
    Otherwise some metrics can't be used. Default: None.")
parser.add_argument("--models", default=None, type=str, help="List of models.")
parser.add_argument("--metric", type=str, help="Chosen fairness metric.")
args = parser.parse_args()

input_file = args.ds
link = args.folder
index = args.index
sens_attrs = ast.literal_eval(args.sensitive)
label = args.label
favored = ast.literal_eval(args.favored)
model_list = ast.literal_eval(args.models)
metric = args.metric
original_data = pd.read_csv("Datasets/" + input_file + ".csv", index_col=index)
dataset = copy.deepcopy(original_data)
original_data = original_data.drop(columns=[label])
for sens in sens_attrs:
    original_data = original_data.loc[:, original_data.columns != sens]

for i, model in enumerate(model_list):
    try:
        original_data_short = pd.read_csv(link + model_list[i] + "_prediction.csv", index_col=index)
        modelnr = i
        break
    except:
        pass
original_data_short = pd.merge(original_data_short, original_data, left_index=True, right_index=True)
original_data_short = original_data_short.loc[:, original_data_short.columns != model_list[modelnr]]
orig_datas = copy.deepcopy(original_data_short)
original_data_short = original_data_short.loc[:, original_data_short.columns != label]
valid_data = dataset.loc[orig_datas.index, :]

try:
    for sens in sens_attrs:
        original_data_short = original_data_short.loc[:, original_data_short.columns != sens]
except:
    original_data_short = original_data_short.loc[:, original_data_short.columns != "sensitive"]
dataset2 = copy.deepcopy(original_data_short)
total_size = len(original_data_short)

groups = dataset[sens_attrs].drop_duplicates(sens_attrs).reset_index(drop=True)
actual_num_of_groups = len(groups)
sensitive_groups = []
sens_cols = groups.columns
for i, row in groups.iterrows():
    sens_grp = []
    for col in sens_cols:
        sens_grp.append(row[col])
    sensitive_groups.append(tuple(sens_grp))

df_count = 0
result_df = pd.DataFrame(columns=["model", "error_rate", "demographic_parity", "equalized_odds", "equal_opportunity", "treatment_equality"])
if metric != "consistency":
    for model in model_list:
        try:
            df = pd.read_csv(link + model + "_prediction.csv", index_col=index)
        except:
            continue

        result_df.at[df_count, "model"] = model

        df = pd.merge(df, original_data_short, left_index=True, right_index=True)

        if actual_num_of_groups == 2:
            ds = StandardDataset(df, 
                label_name=label, 
                favorable_classes=[1], 
                protected_attribute_names=sens_attrs, 
                privileged_classes=[[favored]])

            dataset_pred = ds.copy()
            
            dataset_pred.labels = df[model]

            attr = dataset_pred.protected_attribute_names[0]
            idx = dataset_pred.protected_attribute_names.index(attr)
            privileged_groups =  [{attr:dataset_pred.privileged_protected_attributes[idx][0]}] 
            unprivileged_groups = [{attr:dataset_pred.unprivileged_protected_attributes[idx][0]}] 

            metric_pred = BinaryLabelDatasetMetric(dataset_pred, unprivileged_groups=unprivileged_groups, privileged_groups=privileged_groups)
            class_metric_pred = ClassificationMetric(ds, dataset_pred, unprivileged_groups=unprivileged_groups, privileged_groups=privileged_groups)

            result_df.at[df_count, "error_rate"] = class_metric_pred.error_rate() * 100
            result_df.at[df_count, "demographic_parity"] = abs(metric_pred.statistical_parity_difference()) * 100
            result_df.at[df_count, "equalized_odds"] = abs(class_metric_pred.average_abs_odds_difference()) * 100
            result_df.at[df_count, "equal_opportunity"] = abs(class_metric_pred.equal_opportunity_difference()) * 100
            treq1 = class_metric_pred.num_false_positives(True)/(class_metric_pred.num_false_positives(True) + class_metric_pred.num_false_negatives(True))
            treq2 = class_metric_pred.num_false_positives(False)/(class_metric_pred.num_false_positives(False) + class_metric_pred.num_false_negatives(False))
            result_df.at[df_count, "treatment_equality"] = abs(treq1 - treq2) * 100
            result_df.at[df_count, "tp"] = class_metric_pred.num_true_positives()
            result_df.at[df_count, "tn"] = class_metric_pred.num_true_negatives()
            result_df.at[df_count, "fp"] = class_metric_pred.num_false_positives()
            result_df.at[df_count, "fn"] = class_metric_pred.num_false_negatives()
        else:
            result_df.at[df_count, "model"] = model
            if sens_attrs[0] not in df.columns:
                for sens in sens_attrs:
                    df[sens] = dataset[sens]
            grouped_df = df.groupby(sens_attrs)
            total_ppv = 0
            total_size = 0
            total_ppv_y0 = 0
            total_size_y0 = 0
            total_ppv_y1 = 0
            total_size_y1 = 0
            discr_ppv_y0 = 0
            discr_size_y0 = 0
            discr_ppv_y1 = 0
            discr_size_y1 = 0
            wrong_predicted = 0
            wrong_predicted_y0 = 0
            wrong_predicted_y1 = 0
            total_fp = 0
            total_fn = 0
            num_pos = 0
            num_neg = 0
            dp_grp = []
            eod_grp = []
            eop_grp = []
            treq_grp = []
            #counterfactual = 0
            group_predsize = []
            #Get the favored group to test against and also the averages over the whole dataset
            di = dict()
            for key, item in grouped_df:
                predsize = 0
                part_df = grouped_df.get_group(key)
                for i, row in part_df.iterrows():
                    predsize += 1
                    total_ppv = total_ppv + row[model]
                    total_size = total_size + 1
                    wrong_predicted = wrong_predicted + abs(row[model] - row[label])
                    if row[label] == 0:
                        total_ppv_y0 = total_ppv_y0 + row[model]
                        total_size_y0 = total_size_y0 + 1
                        wrong_predicted_y0 = wrong_predicted_y0 + abs(row[model] - row[label])
                        if row[model] == 1:
                            total_fp += 1
                            num_pos += 1
                        else:
                            num_neg += 1
                    elif row[label] == 1:
                        total_ppv_y1 = total_ppv_y1 + row[model]
                        total_size_y1 = total_size_y1 + 1
                        wrong_predicted_y1 = wrong_predicted_y1 + abs(row[model] - row[label])
                        if row[model] == 0:
                            total_fn += 1
                            num_neg += 1
                        else:
                            num_pos += 1

            result_df.at[df_count, "error_rate"] = wrong_predicted/total_size * 100
            #Iterate again for formula
            count = 0
            dp = 0
            eq_odd = 0
            eq_opp = 0
            tr_eq = 0
            impact = 0
            fp = 0
            fn = 0
            for key, item in grouped_df:
                model_ppv = 0
                model_size = 0
                model_ppv_y0 = 0
                model_size_y0 = 0
                model_ppv_y1 = 0
                model_size_y1 = 0
                part_df = grouped_df.get_group(key)
                for i, row in part_df.iterrows():
                    model_ppv = model_ppv + row[model]
                    model_size = model_size + 1
                    if row[label] == 0:
                        model_ppv_y0 = model_ppv_y0 + row[model]
                        model_size_y0 = model_size_y0 + 1
                        if row[model] == 1:
                            fp += 1
                    elif row[label] == 1:
                        model_ppv_y1 = model_ppv_y1 + row[model]
                        model_size_y1 = model_size_y1 + 1
                        if row[model] == 0:
                            fn += 1

                count += 1

                dp_group = model_ppv/model_size - total_ppv/total_size
                eq_odd_part_group = model_ppv_y0/model_size_y0 - total_ppv_y0/total_size_y0
                eq_opp_group = model_ppv_y1/model_size_y1 - total_ppv_y1/total_size_y1
                eq_odd_group = 0.5*(eq_opp_group) - 0.5*(eq_odd_part_group)
                dp += abs(dp_group)
                
                eq_odd += abs(eq_odd_group)
                eq_opp += abs(eq_opp_group)
                if fp+fn == 0 and total_fp+total_fn == 0:
                    tr_eq_group = 0
                    pass
                elif fp+fn == 0:
                    tr_eq_group = 0.5 - total_fp/(total_fp+total_fn)
                elif total_fp+total_fn == 0:
                    tr_eq_group = fp/(fp+fn) - 0.5
                else:
                    tr_eq_group = fp/(fp+fn) - total_fp/(total_fp+total_fn)
                tr_eq += abs(tr_eq_group)

                #result_df.at[df_count, "dp_" + str(count)] = dp_group
                #result_df.at[df_count, "eop_" + str(count)] = eq_opp_group
                #result_df.at[df_count, "eod_part_" + str(count)] = eq_odd_part_group
                #result_df.at[df_count, "treq_" + str(count)] = tr_eq_group

                dp_grp.append(dp_group)
                eod_grp.append(eq_odd_group)
                eop_grp.append(eq_opp_group)
                treq_grp.append(tr_eq_group)

            result_df.at[df_count, "demographic_parity"] = dp/(len(grouped_df)) * 100
            result_df.at[df_count, "equalized_odds"] = eq_odd/(len(grouped_df)) * 100
            result_df.at[df_count, "equal_opportunity"] = eq_opp/(len(grouped_df)) * 100
            result_df.at[df_count, "treatment_equality"] = tr_eq/(len(grouped_df)) * 100

            result_df.at[df_count, "dp_gap"] = (max(dp_grp) - min(dp_grp)) * 100
            result_df.at[df_count, "eod_gap"] = (max(eod_grp) - min(eod_grp)) * 100
            result_df.at[df_count, "eop_gap"] = (max(eop_grp) - min(eop_grp)) * 100
            result_df.at[df_count, "treq_gap"] = (max(treq_grp) - min(treq_grp)) * 100

        df_count += 1

elif metric == "consistency":
    model_list2 = []
    for model in model_list:
        try:
            pd.read_csv(link + model + "_prediction.csv", index_col="index")
            model_list2.append(model)
        except:
            continue

    df = pd.read_csv(link + model_list2[0] + "_prediction.csv", index_col="index")
    for i, model in enumerate(model_list2):
        if i == 0:
            continue
        df2 = pd.read_csv(link + model + "_prediction.csv", index_col="index")
        df = pd.merge(df, df2[[model]], how="inner", left_index=True, right_index=True)

    #Now evaluate each model according to the metrics implemented.
    model_count = 0
    models_consistency = [0 for model in model_list2]
    #CONSISTENCY TEST, COMPARE PREDICTION TO PREDICTIONS OF NEIGHBORS
    consistency = 0
    for i, row_outer in df.iterrows():
        nbrs = NearestNeighbors(n_neighbors=10, algorithm='kd_tree').fit(dataset2.values)
        indices = nbrs.kneighbors(dataset2.loc[i].values.reshape(1, -1),\
            return_distance=False)
        real_indices = df.index[indices].tolist()
        df_local = df.loc[real_indices[0]]
        model_count = 0
        for model in model_list2:
            knn_ppv = 0
            knn_count = 0
            inacc = 0
            for j, row in df_local.iterrows():
                inacc += abs(row[model] - row[label])
                knn_ppv = knn_ppv + row[model]
                knn_count = knn_count + 1
            knn_pppv = knn_ppv/knn_count
            models_consistency[model_count] = models_consistency[model_count] + abs(df.loc[i][model] - knn_pppv)
            model_count = model_count + 1

    model_count = 0
    for model in model_list2:
        result_df.at[model_count, "model"] = model
        error = 0
        for i, row in df.iterrows():
            error += abs(row[model] - row[label])
        result_df.at[model_count, "error_rate"] = error/len(df) * 100
        result_df.at[model_count, "consistency"] = models_consistency[model_count]/len(df) * 100
        model_count = model_count + 1

result_df.to_csv(link + "EVALUATION_" + str(input_file) + ".csv")

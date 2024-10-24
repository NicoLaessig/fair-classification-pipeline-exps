"""
This code evaluates the results of the experiments based on several metrics.
"""
import copy
from aif360.datasets import StandardDataset
from aif360.metrics import BinaryLabelDatasetMetric, ClassificationMetric
import pandas as pd
from sklearn.neighbors import NearestNeighbors

def acc_bias(dataset_true, preds, unprivileged_groups, privileged_groups, metric="demographic_parity", lam=0.5, no_abs=False):
    dataset_pred = copy.deepcopy(dataset_true)
    dataset_pred.labels = preds

    metric_pred = BinaryLabelDatasetMetric(dataset_pred, unprivileged_groups=unprivileged_groups, privileged_groups=privileged_groups)
    class_metric_pred = ClassificationMetric(dataset_true, dataset_pred, unprivileged_groups=unprivileged_groups, privileged_groups=privileged_groups)

    accuracy = abs(class_metric_pred.accuracy()) * 100
    if not no_abs:
        if metric == "demographic_parity":
            bias = abs(class_metric_pred.statistical_parity_difference()) * 100
        elif metric == "equalized_odds":
            bias = abs(class_metric_pred.average_abs_odds_difference()) * 100
        elif metric == "equal_opportunity":
            bias = abs(class_metric_pred.equal_opportunity_difference()) * 100
        elif metric == "treatment_equality":
            treq1 = class_metric_pred.num_false_positives(True)/(class_metric_pred.num_false_positives(True) + class_metric_pred.num_false_negatives(True))
            treq2 = class_metric_pred.num_false_positives(False)/(class_metric_pred.num_false_positives(False) + class_metric_pred.num_false_negatives(False))
            bias = abs(treq1 - treq2) * 100
        elif metric == "consistency":
            dataset, df_dict = dataset_pred.convert_to_dataframe()
            dataset2 = dataset.loc[:, dataset.columns != df_dict["label_names"][0]]
            models_consistency = 0
            for i, row_outer in dataset2.iterrows():
                nbrs = NearestNeighbors(n_neighbors=10, algorithm='kd_tree').fit(dataset2.values)
                indices = nbrs.kneighbors(dataset2.loc[i].values.reshape(1, -1),\
                    return_distance=False)
                real_indices = dataset2.index[indices].tolist()
                df_local = dataset.loc[real_indices[0]]
                model_count = 0
                knn_ppv = 0
                knn_count = 0
                inacc = 0
                for j, row in df_local.iterrows():
                    knn_ppv = knn_ppv + row[df_dict["label_names"][0]]
                    knn_count = knn_count + 1   
                knn_pppv = knn_ppv/knn_count
                models_consistency += abs(dataset.loc[i][df_dict["label_names"][0]] - knn_pppv)
            bias = models_consistency/len(dataset)*100
        
        return 100 - (lam*(100-accuracy) + (1-lam)*bias)


    else:
        if metric == "demographic_parity":
            bias = class_metric_pred.statistical_parity_difference()
        elif metric == "equalized_odds":
            bias = class_metric_pred.average_abs_odds_difference()
        elif metric == "equal_opportunity":
            bias = class_metric_pred.equal_opportunity_difference()
        elif metric == "treatment_equality":
            treq1 = class_metric_pred.num_false_positives(True)/(class_metric_pred.num_false_positives(True) + class_metric_pred.num_false_negatives(True))
            treq2 = class_metric_pred.num_false_positives(False)/(class_metric_pred.num_false_positives(False) + class_metric_pred.num_false_negatives(False))
            bias = treq1 - treq2
        
        return bias

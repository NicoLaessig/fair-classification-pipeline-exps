from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import copy
import falconn

from sklearn.metrics.pairwise import euclidean_distances
from sklearn.metrics import roc_auc_score
from aif360.metrics import ClassificationMetric
import matplotlib.pyplot as plt


def measure_error(label, edge, w_edge):
    """         
        Computes the total error in the data.
        Args:
            label: Labels of the data
            edge: Indices of similar pairs
            w_edge: Similarity values of similar pairs
        Return:
            total_error: The total error
    """

    total_error = np.sum(w_edge * abs(label[edge[:,0]]-label[edge[:,1]]))

    return total_error

def measure_consistency(model, data, edge, w_edge):
    """         
        Computes the consistency score, 1 - sum (|yi_hat - yj_hat| * Wij) / sum (Wij)

        Args:
            model: Trained model
            data: Features of the data
            w_sim: Similarity matrix
            edge: Indices of similar pairs

        Return:
            consistency score: Consistency score on the data
    """
    
    y_hat = model.predict(data)
    
    total_error = measure_error(y_hat, edge, w_edge)
    num_similar_pairs = len(edge)
    
    consistency_score = 1 - total_error / num_similar_pairs
    return consistency_score

def generate_sim_matrix(data, similarity_matrix, similarity_params):
    """         
        Obtains approximate kNN-based/threshold-based similarity matrix and similar pairs using LSH.
            kNN-based: Considers (xi, xj) as a similar pair if xi is one of xj's nearest neighbors or vice versa.
            threshold-based: Considers (xi, xj) as a similar pair if their distance is smaller than a threshold.

        Args:
            data: Features of the data
            similarity_matrix: "knn" or "threshold" based similarity matrix
            similarity_params: Hyperparameters for similarity matrix

        Return:
            w_sim: Similarity matrix
            edge: Indices of similar pairs
            w_edge: Similarity values of similar pairs
    """

    n, d = data.shape
    w_sim = np.zeros((n, n))

    params = falconn.get_default_parameters(n, d)
    params.lsh_family = falconn.LSHFamily.Hyperplane

    params.k = similarity_params["num_hash"]
    params.l = similarity_params["num_table"]

    table = falconn.LSHIndex(params)
    table.setup(data)
    qo = table.construct_query_object()
    
    edge = []
    for i in range(data.shape[0]):
        if similarity_matrix == "knn":
            indices = qo.find_k_nearest_neighbors(data[i], similarity_params["k"]+1)
        elif similarity_matrix == "threshold":
            indices = qo.find_near_neighbors(data[i], similarity_params["threshold"]**2)

        distances = np.squeeze(euclidean_distances([data[i]], data[indices]))
        w_sim[i, indices] = np.exp(-1 * similarity_params["theta"] * distances)
        w_sim[indices, i] = np.exp(-1 * similarity_params["theta"] * distances)
        w_sim[i][i] = 0

    temp = np.argwhere(w_sim > 0)
    edge = np.squeeze(temp[np.argwhere(temp[:, 0] < temp[:, 1])])
    w_edge = w_sim[edge[:,0], edge[:,1]]

    return w_sim, edge, w_edge


def generate_original_sim_matrix(data, similarity_matrix, similarity_params):
    """         
        Obtains exact kNN-based/threshold-based similarity matrix and similar pairs.
            kNN-based: Considers (xi, xj) as a similar pair if xi is one of xj's nearest neighbors or vice versa.
            threshold-based: Considers (xi, xj) as a similar pair if their distance is smaller than a threshold.

        Args:
            data: Features of the data
            similarity_matrix: "knn" or "threshold" based similarity matrix
            similarity_params: Hyperparameters for similarity matrix

        Return:
            w_sim: Similarity matrix
            edge: Indices of similar pairs
            w_edge: Similarity values of similar pairs
    """

    n, d = data.shape
    w_sim = np.zeros((n, n))
    
    for i in range(data.shape[0]):
        distances = np.squeeze(euclidean_distances([data[i]], data))
        if similarity_matrix == "knn":
            idx = np.argsort(distances)[:similarity_params["k"]+1]
        elif similarity_matrix == "threshold":
            idx = distances < similarity_params["threshold"]

        w_sim[i, idx] = 1
        w_sim[idx, i] = 1
        w_sim[i][i] = 0
        
    temp = np.argwhere(w_sim > 0)
    edge = np.squeeze(temp[np.argwhere(temp[:, 0] < temp[:, 1])])
    w_edge = w_sim[edge[:,0], edge[:,1]]
                
    return w_sim, edge, w_edge


def performance_func(model, data, label, edge, w_edge):
    """         
        Returns model's accuracy and consistency score on the data.

        Args:
            model: Trained model
            data: Features of the data
            label: Labels of the data
            edge: Indices of similar pairs
            w_edge: Similarity values of similar pairs

        Return:
            accuracy: Model's accuracy on the data
            consistency score: Consistency score on the data
    """

    accuracy = model.score(data, label)
    consistency_score = measure_consistency(model, data, edge, w_edge)
    
    return accuracy, consistency_score


def plot_result(opt, m_list, performance_arr, y_axis, num_digits, target = "_", num_figures = 1):
    labels = np.array(m_list)[::-1]

    plot = list()
    for k in performance_arr:
        plot.append(np.array(performance_arr[k])[::-1])

    x = np.arange(len(labels)) # the label locations
    width = 0.17  # the width of the bars

    plt.figure(num_figures, figsize=(20,15))
    ax = plt.subplot()
    [x.set_linewidth(2) for x in ax.spines.values()]

    rts = list()
    color_list = ["pink", "lightblue", "thistle", "darkgray", "wheat"]
    hatch_list = ['//', '\\', '/', 'X', '']
    ablation_color_dict = {1:0, 0:1, 3:2, 4:3, 2:4}
    ablation_hatch_dict = {2:0, 1:1, 3:2, 4:3, 0:4}
    if target=="ablation":
        color_list = sorted(color_list, key=lambda x:ablation_color_dict[color_list.index(x)])
        hatch_list = sorted(hatch_list, key=lambda x:ablation_hatch_dict[hatch_list.index(x)])

    for i, e in enumerate(performance_arr):
        rts.append(ax.barh(x + width * (2 - i), np.round(plot[i], num_digits), width, label=e, color=color_list[i], hatch=hatch_list[i],edgecolor="black", linewidth=2))

    # Add some text for labels, title and custom x-axis tick labels, etc.
    plt.tick_params(labelsize=40)
    ax.set_xlabel(y_axis, fontsize=45)
    ax.set_ylabel("Total Error Limit (m)", fontsize=45)

    ax.set_yticks(x)
    ax.set_yticklabels(labels)
    max_val, max_ratio = 0, 1.06
    left, right = plt.xlim()
    # xmax = 10**(np.ceil(np.log10(right)*1.05))
    xmax = 10**(np.log10(right)*1.20)
    plt.xlim([1.0, xmax])

    # ax.set_xscale('log')
    # plt.legend(prop={'size':30}, bbox_to_anchor=(-0.025, 1), loc="lower left", ncol=5)
    # plt.legend(prop={'size':30}, loc="upper right", ncol=1)
    if target=="ablation":
        ax.set_xscale('log')
        if y_axis == "Total Error":
            plt.legend(prop={'size':30}, loc="lower right", ncol=1)
    elif target=="solution":
        ax.set_xscale('log')
        if y_axis == "# Flips":
            plt.legend(prop={'size':30}, loc="upper right", ncol=1)
            xmax = 10**(np.log10(right)+1.8)
            plt.xlim([1.0, xmax])
        
    def autolabel(rects):
        """Attach a text label above each bar in *rects*, displaying its height."""
        for rect in rects:
            ax.annotate('{}'.format(rect.get_width()),
                        xy=(max(rect.get_width(), max_val)*max_ratio, rect.get_y() + rect.get_height()*0.285),
                        fontsize=35)

    for rt in rts:
        autolabel(rt)

    plt.tight_layout()
    # plt.show()
    name = y_axis.replace(" ","")
    plt.savefig(f"{opt.save_directory}/{target}_comparison_{name}_{opt.dataset}_{opt.similarity_matrix}_{opt.model_type}.png")
    print(f"{name}_{plt.axis()}")

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import copy

from sklearn.cluster import KMeans 
try:
    from .utils import measure_error
except:
    from utils import measure_error
    
def kMeans(data, label, m, edge, w_edge):
    """         
        Applies k-means clustering and, for each cluster, make its examples have the majority label.
        Here one would have to adjust k to find the clusters with just the right amount of total error m.

        Args: 
            data: Features of the data
            label: Labels of the data
            m: The total error limit
            edge: Indices of similar pairs
            w_edge: Similarity values of similar pairs
            
        Return:
            flipped_label: Flipped labels for a given m
    """

    k_high, k_low = min(label.shape[0], 20000), 1 # set upper limit

    init_error = measure_error(label, edge, w_edge)
    best_flips, best_error = label.shape[0], init_error
    best_flips_fail, best_error_fail = label.shape[0], init_error

    while (k_high - k_low > 1):
        k = int((k_high + k_low) / 2)

        kmeans_label = copy.deepcopy(label)
        kmeans = KMeans(n_clusters=k, random_state=0).fit(data)
        for i in range(k):
            idx = (kmeans.labels_ == i)
            labels, num_labels = np.unique(label[idx], return_counts=True)

            if len(num_labels) > 0:
                max_idx = np.argmax(num_labels)
                major_label = int(labels[max_idx])
                kmeans_label[idx] = major_label

        total_error = measure_error(kmeans_label, edge, w_edge)
        num_flips = np.sum(label != kmeans_label)

        if total_error <= m: 
            if num_flips < best_flips:
                best_flips = num_flips
                best_error = total_error
                best_flipped_label = copy.deepcopy(kmeans_label)
            k_low = k
        else:
            if (best_error > m) and (total_error < best_error_fail):
                best_flips_fail = num_flips
                best_error_fail = total_error
                best_flipped_label_fail = copy.deepcopy(kmeans_label)
            k_high = k

        if best_error <= m:
            flipped_label = copy.deepcopy(best_flipped_label)
        else:
            flipped_label = copy.deepcopy(best_flipped_label_fail)

    return flipped_label
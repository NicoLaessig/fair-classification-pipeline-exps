from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import copy

try:
    from .utils import measure_error
except:
    from utils import measure_error
    
def Greedy(label, m, w_sim, edge, w_edge):
    """         
        Flips labels that reduce the total error the most.
        
        Args: 
            label: Labels of the data
            m: The total error limit
            w_sim: Similarity matrix
            edge: Indices of similar pairs
            w_edge: Similarity values of similar pairs
            
        Return:
            flipped_label: Flipped labels for a given m
    """

    flipped_label = copy.deepcopy(label)

    total_error = measure_error(flipped_label, edge, w_edge)
    prev_error = total_error
    while total_error > m:
        max_indices, improvement_arr = [], []
        total_improvement = 0
        
        for j in range(len(flipped_label)):
            prev_y_diff = (flipped_label != flipped_label[j])

            improvement = np.sum(w_sim[j] * (2 * prev_y_diff - 1))
            improvement_arr.append(improvement)

        sorted_index = np.argsort(improvement_arr)[::-1][:1]
        for index in sorted_index:
            if improvement_arr[index] <= 0:
                break

            total_improvement += improvement_arr[index]
            max_indices.append(index)
            if total_improvement >= (total_error - m):
                break

        flipped_label[max_indices] = 1-flipped_label[max_indices]
        total_error = measure_error(flipped_label, edge, w_edge)

        # Until there is no more error reduction
        if prev_error <= total_error:
            print("no more reduction")
            break
        else:
            prev_error = total_error
            
    return flipped_label
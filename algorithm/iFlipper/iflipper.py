from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import copy

try:
    from .utils import measure_error
except:
    from utils import measure_error
try:
    from .mosek_solver import MOSEK_Solver
except:
    from mosek_solver import MOSEK_Solver
try:
    from .iflipper_utils import init_cluster, get_zero_cluster, get_nonzero_two_clusters, transform_with_one_cluster, transform_with_two_clusters
except:
    from iflipper_utils import init_cluster, get_zero_cluster, get_nonzero_two_clusters, transform_with_one_cluster, transform_with_two_clusters

class iFlipper:
    def __init__(self, label, w_sim, edge, w_edge, ablation_option = None):
        """         
            Args: 
                label: Labels of the data
                w_sim: Similarity matrix
                edge: Indices of similar pairs
                w_edge: Similarity values of similar pairs
        """

        self.label = label
        self.w_sim = w_sim
        self.edge = edge
        self.w_edge = w_edge
        self.ablation_option = ablation_option

    def transform(self, m):
        """         
            Solves the optimization problem of minimally flipping labels given a limit to the total error limit m.

            Args: 
                m: The total error limit
                
            Return:
                target_label: Flipped labels for a given m
        """

        if measure_error(self.label, self.edge, self.w_edge) == m:
            target_label = self.label
        elif self.ablation_option == None or self.ablation_option == "iFlipper":
            optimal_label = MOSEK_Solver(self.label, m, self.w_sim, self.edge)
            converted_label = self.converting_solution(optimal_label, self.label, self.w_sim, self.edge)
            rounded_label = self.adaptive_rounding(converted_label, m, self.w_sim, self.edge)
            flipped_label = self.reverse_greedy(self.label, rounded_label, m, self.w_sim, self.edge, self.w_edge)
            target_label = flipped_label

        elif self.ablation_option == "LP-SR":
            optimal_label = MOSEK_Solver(self.label, m, self.w_sim, self.edge)
            converted_label = self.converting_solution(optimal_label, self.label, self.w_sim, self.edge)
            target_label = np.round(converted_label)

        elif self.ablation_option == "LP-AR":
            optimal_label = MOSEK_Solver(self.label, m, self.w_sim, self.edge)
            converted_label = self.converting_solution(optimal_label, self.label, self.w_sim, self.edge)
            rounded_label = self.adaptive_rounding(converted_label, m, self.w_sim, self.edge)
            target_label = rounded_label

        return target_label

    def converting_solution(self, optimal_label, label, w_sim, edge):
        """         
            Converts an optimal solution for the LP problem to another optimal solution whose values are in {0, alpha, 1}.

            Args: 
                optimal_label: Optimal solution for the LP problem
                label: Labels of the data
                w_sim: Similarity matrix
                edge: Indices of similar pairs
                
            Return:
                converted_label: Transformed optimal solution where each value is one of {0, alpha, 1}
        """

        converted_label = copy.deepcopy(optimal_label)
        unique_values = np.unique(converted_label)
        idx = (unique_values >= 1e-7) & (unique_values <= 1-1e-7)
        
        if np.sum(idx) >= 2:
            cluster_info, cluster_nodes, cluster_nodes_num = init_cluster(optimal_label, label, w_sim, edge)
            T = len(cluster_info.keys())
            
            while T > 1:
                while True:
                    alpha = get_zero_cluster(cluster_nodes_num)
                    if alpha == 0:
                        break
                    else:
                        cluster_info, cluster_nodes_num, cluster_nodes = transform_with_one_cluster(alpha, cluster_info, cluster_nodes_num, cluster_nodes)
                        T = len(cluster_info.keys())

                if T > 1:
                    alpha, beta = get_nonzero_two_clusters(cluster_info)
                    cluster_info, cluster_nodes_num, cluster_nodes = transform_with_two_clusters(alpha, beta, cluster_info, cluster_nodes_num, cluster_nodes)    
                    T = len(cluster_info.keys())

            for value in cluster_nodes:
                for node in cluster_nodes[value]:
                    converted_label[node] = value
            
        return converted_label

    def adaptive_rounding(self, converted_label, m, w_sim, edge):
        """         
            Converts a transformed optimal solution into a feasible solution whose values are only 0's ans 1's.

            Args: 
                opt_solution: Optimal solution for the LP problem where each value is one of {0, alpha, 1}
                m: The total error limit
                w_sim: Similarity matrix
                edge: Indices of similar pairs
                
            Return:
                rounded_label: Feasible binary integer solution
        """

        rounded_label = copy.deepcopy(converted_label)
        unique_values = np.unique(rounded_label)
        idx = (unique_values >= 1e-7) & (unique_values <= 1-1e-7)

        if np.sum(idx) >= 1:
            N0, N1 = 0, 0
            for (i, j) in edge:
                violation = abs(rounded_label[i] - rounded_label[j])
                if 1e-7 <= violation <= 1 - 1e-7:
                    if rounded_label[i] <= 1e-7 or rounded_label[j] <= 1e-7:
                        N0 += w_sim[i][j]
                    elif rounded_label[i] >= 1-1e-7 or rounded_label[j] >= 1-1e-7:
                        N1 += w_sim[i][j]

            index = (1e-7 < rounded_label) & (rounded_label < 1-1e-7)
            if N0 <= N1:
                rounded_label[index] = 1
            else:
                rounded_label[index] = 0

        return np.round(rounded_label)

    def reverse_greedy(self, original_label, rounded_label, m, w_sim, edge, w_edge):
        """         
            Unflips labels that increase the total error the least.

            Args: 
                original_label: Original labels
                rounded_label: Feasible binary integer solution from the adaptive rounding algorithm
                m: The total error limit
                w_sim: Similarity matrix
                edge: Indices of similar pairs
                w_edge: Similarity values of similar pairs
                
            Return:
                flipped_label: Flipped labels for a given m
        """

        flipped_label = copy.deepcopy(rounded_label)
        flipped_bool = (original_label != rounded_label)
        flipped_index = np.arange(len(rounded_label))[flipped_bool]

        total_error = measure_error(rounded_label, edge, w_edge)
        while total_error < m:
            flipped_label = copy.deepcopy(rounded_label)
            improvement_arr = []
            for j in range(len(rounded_label)):
                if flipped_bool[j]:
                    prev_y_diff = (rounded_label != rounded_label[j])
                    negative_improvement = - np.sum(w_sim[j] * (2 * prev_y_diff - 1))
                    improvement_arr.append(negative_improvement)

            sorted_index = np.argsort(improvement_arr)[:1]

            rounded_label[flipped_index[sorted_index]] = 1 - rounded_label[flipped_index[sorted_index]]
            flipped_bool[flipped_index[sorted_index]] = False
            flipped_index = np.arange(len(rounded_label))[flipped_bool]

            total_error = measure_error(rounded_label, edge, w_edge)

        return flipped_label
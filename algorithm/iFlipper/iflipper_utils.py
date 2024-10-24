from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import copy

def init_cluster(optimal_label, label, w_sim, edge):
    """         
        Initializes cluster information

        Args: 
            optimal_label: Optimal solution for the LP problem
            label: Labels of the data
            w_sim: Similarity matrix
            edge: Indices of similar pairs

        Return: 
            cluster_info: Contains information about the connected clusters for each cluster
            cluster_nodes: Contains a collection of nodes in the cluster for each cluster
            cluster_nodes_num: Contains N_alpha value (Number of intial label 0's - 1's in the cluster) for each cluster
    """

    cluster_info, cluster_nodes, cluster_nodes_num = dict(), dict(), dict()
    for i in range(len(optimal_label)):
        value = optimal_label[i]
        if value not in cluster_nodes:
            cluster_nodes[value] = [i]
        else:
            cluster_nodes[value].append(i)

        if (0 < value < 1):
            if value not in cluster_nodes_num:
                cluster_nodes_num[value] = (1-2*label[i])
            else:
                cluster_nodes_num[value] += (1-2*label[i])

            if value not in cluster_info:
                cluster_info[value] = dict()
    
    for (i, j) in edge:
        valuei, valuej, sim_value = optimal_label[i], optimal_label[j], w_sim[i][j]
        if valuei != valuej:
            cluster_info = init_cluster_info(valuei, valuej, sim_value, cluster_info)
            cluster_info = init_cluster_info(valuej, valuei, sim_value, cluster_info)

    return cluster_info, cluster_nodes, cluster_nodes_num

def init_cluster_info(valuei, valuej, sim_value, cluster_info):
    """         
        Constructs cluster_info for a similar pair.

        Args: 
            valuei: Optimal label for yi, which is non-0/1 value
            valuej: Optimal label for yj, which is non-0/1 value
            sim_value: Similarity value between ith and jth nodes
            cluster_info: Contains information about the connected clusters for each cluster

        Return:
            cluster_info: Updated cluster_info using a similar pair
    """
    if (0 < valuei < 1):
        if valuej not in cluster_info[valuei]:
            cluster_info[valuei][valuej] = sim_value
        else:
            cluster_info[valuei][valuej] += sim_value

    return cluster_info

def transform_with_one_cluster(alpha, cluster_info, cluster_nodes_num, cluster_nodes):
    """         
        For an alpha-cluster with N_alpha=0, it converts alpha to either a_k or a_k+1 while maintaining an optimal solution (Lemma 2.1 in the paper).

        Args: 
            alpha: Initial value of an alpha-cluster
            cluster_info: Contains information about the connected clusters for each cluster
            cluster_nodes: Contains a collection of nodes in the cluster for each cluster
            cluster_nodes_num: Contains N_alpha value (Number of intial label 0's - 1's in the cluster) for each cluster

        Return: 
            cluster_info: Updated cluster_info after the conversion
            cluster_nodes: Updated cluster_nodes after the conversion
            cluster_nodes_num: Updated cluster_nodes_num after the conversion
    """

    ak, ak1, sum_less_than_alpha, sum_greater_than_alpha = get_cluster_info(alpha, cluster_info)
    if sum_less_than_alpha - sum_greater_than_alpha <= 0:
        alpha_ = ak1
    else:
        alpha_ = ak
    
    cluster_info, cluster_nodes_num, cluster_nodes = update_cluster_info(alpha, alpha_, cluster_info, cluster_nodes_num, cluster_nodes)

    return cluster_info, cluster_nodes_num, cluster_nodes 

def get_zero_cluster(cluster_nodes_num):
    """         
        Obtains an alpha-cluster with N_alpha=0 in the solution.

        Args: 
            cluster_nodes_num: Contains N_alpha value (Number of intial label 0's - 1's in the cluster) for each cluster

        Return: 
            alpha: Value of an alpha-cluster
    """

    alpha = 0
    for value in cluster_nodes_num:
        if cluster_nodes_num[value] == 0:
            alpha = value
            break

    return alpha

def get_cluster_info(alpha, cluster_info):
    """         
        Obtains information about nodes connected to an alpha-cluster.

        Args: 
            alpha: Value of an alpha-cluster
            cluster_info: Contains information about the connected clusters for each cluster

        Return: 
            ak: The largest value among nodes connected to the alpha-cluster whose values are less than the alpha value
            ak1: The smallest value among nodes connected to the alpha-cluster whose values are greater than the alpha value
            sum_less_than_alpha: The total similarity values between the alpha-cluster and the connected nodes whose values are less than the alpha value
            sum_greater_than_alpha: The total similarity values between the alpha-cluster and the connected nodes whose values are greater than the alpha value
    """

    sum_less_than_alpha, sum_greater_than_alpha = 0, 0
    ak, ak1= 0, 1
    for connected_node in cluster_info[alpha]:
        if connected_node < alpha:
            sum_less_than_alpha += cluster_info[alpha][connected_node]
        else:
            sum_greater_than_alpha += cluster_info[alpha][connected_node]

        if ak < connected_node and connected_node < alpha:
            ak = connected_node
        elif ak1 > connected_node and connected_node > alpha:
            ak1 = connected_node
            
    return ak, ak1, sum_less_than_alpha, sum_greater_than_alpha

def transform_with_two_clusters(alpha, beta, cluster_info, cluster_nodes_num, cluster_nodes):
    """         
        For an alpha-cluster with N_alpha!=0 and a beta-cluster with N_beta!=0, it converts (alpha, beta) to one of the five cases in Lemma 2.2 in the paper.

        Args: 
            alpha: Initial value of an alpha=cluster
            beta: Initial value of a beta-cluster
            cluster_info: Contains information about the connected clusters for each cluster
            cluster_nodes: Contains a collection of nodes in the cluster for each cluster
            cluster_nodes_num: Contains N_alpha value (Number of intial label 0's - 1's in the cluster) for each cluster

        Return: 
            cluster_info: Updated cluster_info after the conversion
            cluster_nodes: Updated cluster_nodes after the conversion
            cluster_nodes_num: Updated cluster_nodes_num after the conversion
    """

    ak, ak1, N_alpha, sum_less_than_alpha, sum_greater_than_alpha, bl, bl1, N_beta, sum_less_than_beta, sum_greater_than_beta, sum_btw_clusters = get_clusters_info(alpha, beta, cluster_info, cluster_nodes_num)
    X, Y = N_alpha / N_beta, ((sum_less_than_alpha-sum_greater_than_alpha-sum_btw_clusters)*N_beta - (sum_less_than_beta-sum_greater_than_beta+sum_btw_clusters)*N_alpha)/N_beta
    
    if X < 0 and Y <= 0:
        if X + 1 <=0:
            epsilon_alpha = min(ak1-alpha, -(N_beta/N_alpha)*(bl1-beta))
        else:
            epsilon_alpha = min(ak1-alpha, -(N_beta/N_alpha)*(bl1-beta), (N_beta*(beta-alpha))/(N_alpha+N_beta))
        alpha_ = np.round(alpha + epsilon_alpha, decimals=5)
        beta_ = np.round(beta - (N_alpha / N_beta) * epsilon_alpha, decimals=5)
        
    elif X < 0 and Y > 0:
        if X + 1 >= 0: 
            epsilon_alpha = min(alpha-ak, -(N_beta/N_alpha)*(beta-bl))
        else:
            epsilon_alpha = min(alpha-ak, -(N_beta/N_alpha)*(beta-bl), -(N_beta*(beta-alpha))/(N_alpha+N_beta))
        alpha_ = np.round(alpha - epsilon_alpha, decimals=5)
        beta_ = np.round(beta + (N_alpha / N_beta) * epsilon_alpha, decimals=5)

    elif X > 0 and Y <= 0:
        epsilon_alpha = min(ak1-alpha, (N_beta/N_alpha)*(beta-bl), (N_beta*(beta-alpha))/(N_alpha+N_beta))
        alpha_ = np.round(alpha + epsilon_alpha, decimals=5)
        beta_ = np.round(beta - (N_alpha / N_beta) * epsilon_alpha, decimals=5)

    elif X > 0 and Y > 0:
        epsilon_alpha = min(alpha-ak, (N_beta/N_alpha)*(bl1-beta))
        alpha_ = np.round(alpha - epsilon_alpha, decimals=5)
        beta_ = np.round(beta + (N_alpha / N_beta) * epsilon_alpha, decimals=5)

    total = 0
    for value in cluster_nodes:
        total += len(cluster_nodes[value])

    cluster_info, cluster_nodes_num, cluster_nodes = update_cluster_info(alpha, alpha_, cluster_info, cluster_nodes_num, cluster_nodes)
    cluster_info, cluster_nodes_num, cluster_nodes = update_cluster_info(beta, beta_, cluster_info, cluster_nodes_num, cluster_nodes)

    return cluster_info, cluster_nodes_num, cluster_nodes

def get_nonzero_two_clusters(cluster_info):
    """         
        Obtains an alpha-cluster and a beta-cluster in the solution

        Args: 
            cluster_info: Contains information about the connected clusters for each cluster

        Return: 
            alpha: Value of an alpha-cluster
            beta: Value of a beta-cluster
    """

    value1 = 0
    for value in cluster_info:
        if value1 != 0:
            value2 = value
            break
        else:
            value1 = value

    alpha, beta = min(value1, value2), max(value1, value2)
    return alpha, beta

def get_clusters_info(alpha, beta, cluster_info, cluster_nodes_num):
    """         
        Obtains information about nodes connected to an alpha-cluster and a beta-cluster.

        Args: 
            alpha: Value of an alpha-cluster
            beta: Initial value of a beta-cluster
            cluster_info: Contains information about the connected clusters for each cluster
            cluster_nodes_num: Contains N_alpha value (Number of intial label 0's - 1's in the cluster) for each cluster

        Return: 
            ak: The largest value among nodes connected to the alpha-cluster whose values are less than the alpha value and are not beta value
            ak1: The smallest value among nodes connected to the alpha-cluster whose values are greater than the alpha value and are not beta value
            N_alpha: N_alpha value of the alpha-cluster
            sum_less_than_alpha: The total similarity values between the alpha-cluster and the connected nodes whose values are less than the alpha value
            sum_greater_than_alpha: The total similarity values between the alpha-cluster and the connected nodes whose values are greater than the alpha value
            bl: The largest value among nodes connected to the beta-cluster whose values are less than the beta value and are not alpha value
            bl1: The smallest value among nodes connected to the beta-cluster whose values are greater than the beta value and are not alpha value
            N_beta: N_beta value of the beta-cluster
            sum_less_than_beta: The total similarity values between the beta-cluster and the connected nodes whose values are less than the beta value
            sum_lower_higher_beta: The total similarity values between the beta-cluster and the connected nodes whose values are greater than the beta value
            sum_btw_clusters: The total similarity values between the two clusters
    """

    sum_less_than_alpha, sum_greater_than_alpha, sum_less_than_beta, sum_lower_higher_beta, sum_btw_clusters = 0, 0, 0, 0, 0
    ak, ak1, bl, bl1 = 0, 1, 0, 1
    N_alpha, N_beta = cluster_nodes_num[alpha], cluster_nodes_num[beta]

    for connected_node in cluster_info[alpha]:
        if connected_node < alpha:
            sum_less_than_alpha += cluster_info[alpha][connected_node]
        else:
            sum_greater_than_alpha += cluster_info[alpha][connected_node]

        if connected_node != beta:
            if ak < connected_node and connected_node < alpha:
                ak = connected_node
            elif ak1 > connected_node and connected_node > alpha:
                ak1 = connected_node
        else:
            sum_btw_clusters = cluster_info[alpha][connected_node]
            sum_greater_than_alpha = sum_greater_than_alpha - sum_btw_clusters
        
    for connected_node in cluster_info[beta]:
        if connected_node < beta:
            sum_less_than_beta += cluster_info[beta][connected_node]
        else:
            sum_lower_higher_beta += cluster_info[beta][connected_node]
            
        if connected_node != alpha:
            if bl < connected_node and connected_node < beta:
                bl = connected_node
            elif bl1 > connected_node and connected_node > beta:
                bl1 = connected_node
        else:
            sum_less_than_beta = sum_less_than_beta - sum_btw_clusters

    return ak, ak1, N_alpha, sum_less_than_alpha, sum_greater_than_alpha, bl, bl1, N_beta, sum_less_than_beta, sum_lower_higher_beta, sum_btw_clusters

def update_cluster_info(alpha, alpha_, cluster_info, cluster_nodes_num, cluster_nodes):
    """         
        Updates related cluster information after converting alpha to alpha_ in the solution.

        Args: 
            alpha: Initial value of the cluster
            alpha_: Converted value of the cluster
            cluster_info: Contains information about the connected clusters for each cluster
            cluster_nodes: Contains a collection of nodes in the cluster for each cluster
            cluster_nodes_num: Contains N_alpha value (Number of intial label 0's - 1's in the cluster) for each cluster

        Return: 
            cluster_info: Updated cluster_info after the conversion
            cluster_nodes: Updated cluster_nodes after the conversion
            cluster_nodes_num: Updated cluster_nodes_num after the conversion
    """

    if alpha != alpha_:
        for connected_node in cluster_info[alpha]:
            if (0 < connected_node < 1):
                if alpha_ not in cluster_info[connected_node]:
                    cluster_info[connected_node][alpha_] = cluster_info[connected_node][alpha]

                del cluster_info[connected_node][alpha]

        if (0 < alpha_< 1):
            if alpha_ not in cluster_info:
                cluster_info[alpha_] = cluster_info[alpha]
                cluster_nodes_num[alpha_] = cluster_nodes_num[alpha]
            else:
                cluster_nodes_num[alpha_] += cluster_nodes_num[alpha]
                for connected_node in cluster_info[alpha]:
                    if connected_node not in cluster_info[alpha_]:
                        cluster_info[alpha_][connected_node] = cluster_info[alpha][connected_node]
                    else:
                        cluster_info[alpha_][connected_node] += cluster_info[alpha][connected_node]

            if alpha_ in cluster_info[alpha_]:
                del cluster_info[alpha_][alpha_]

        if alpha_ not in cluster_nodes:
            cluster_nodes[alpha_] = cluster_nodes[alpha]
        else:
            cluster_nodes[alpha_].extend(cluster_nodes[alpha])

        del cluster_info[alpha]
        del cluster_nodes_num[alpha]
        del cluster_nodes[alpha]

    return cluster_info, cluster_nodes_num, cluster_nodes

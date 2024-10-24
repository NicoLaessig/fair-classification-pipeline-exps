from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import time
import cplex

def CPLEX_Solver(label, m, w_sim, edge, ILP = True):
    """         
        Solves the label flipping optimization problem for a given total error limit m using CPLEX.
        If ILP is True, it solves the ILP problem, otherwise solves the LP problem.

        Args: 
            label: Labels of the data
            m: The total error limit
            w_sim: Similarity matrix
            edge: Indices of similar pairs
            ILP: Indicates the type of problem
            
        Return:
            flipped_label: Flipped labels for a given m
    """

    prob, x_idx = problem_setup(label, w_sim, edge, ILP)
    
    # Set the total error limit m 
    total_error_limit = get_error(prob)
    prob.linear_constraints.set_rhs(total_error_limit, m)

    # Solve the problem
    prob.solve()
    sol = prob.solution

    return np.round(sol.get_values(list(x_idx)), decimals=4)

def problem_setup(label, w_sim, edge, ILP):
    """         
        Constructs the label flipping optimization problem.
        If ILP is True, it constructs the ILP problem, otherwise constructs the LP problem.

        Args: 
            label: Labels of the data
            w_sim: Similarity matrix
            edge: Indices of similar pairs
            ILP: Indicates the type of problem
            
        Return:
            prob: Optimization problem
            x_idx: Index of the solution
    """

    m = 0
    prob = cplex.Cplex()
    prob.set_log_stream(None)
    prob.set_error_stream(None)
    prob.set_warning_stream(None)
    prob.set_results_stream(None)
    prob.objective.set_sense(prob.objective.sense.minimize)

    name_z = []
    name_x = []
    for i in range(label.shape[0]):
        name_z.append(f'z_{i}')
        name_x.append(f'x_{i}')
        
    prob.variables.add(names=name_z,obj = [1.0] * label.shape[0],types = [prob.variables.type.continuous] * label.shape[0], lb=[0.0] * label.shape[0], ub=[1.0]* label.shape[0])
    
    if ILP:
        x_idx = prob.variables.add(names=name_x,types = [prob.variables.type.binary] * label.shape[0], lb=[0] * label.shape[0], ub=[1]* label.shape[0])
    else:
        x_idx = prob.variables.add(names=name_x,types = [prob.variables.type.continuous] * label.shape[0], lb=[0] * label.shape[0], ub=[1]* label.shape[0])

    for i in range(label.shape[0]):
        prob.linear_constraints.add(
            lin_expr = [cplex.SparsePair(ind = [f"x_{i}", f"z_{i}"], val = [-1.0, 1.0]),
                        cplex.SparsePair(ind = [f"x_{i}", f"z_{i}"], val = [-1.0, 1.0]),
                        cplex.SparsePair(ind = [f"x_{i}", f"z_{i}"], val = [1.0, 1.0]),
                        cplex.SparsePair(ind = [f"x_{i}", f"z_{i}"], val = [1.0, 1.0])],
            senses = ["L", "G", "G", "L"],
            rhs = [label[i], -label[i], label[i], 2-label[i]],
            names = [f'c_{i}0',f'c_{i}1',f'c_{i}2',f'c_{i}3'])

    prob.linear_constraints.add(names=['violation'], senses=["L"], rhs=[m])
    for (i, j) in edge:
        prob.variables.add(names=[f'z_{i}{j}'], types = [prob.variables.type.continuous],lb=[0], ub=[1])
        prob.linear_constraints.set_linear_components('violation', [[f"z_{i}{j}"], [w_sim[i][j]]])
        prob.linear_constraints.add(
            lin_expr = [cplex.SparsePair(ind = [f"x_{i}", f"x_{j}",f"z_{i}{j}"], val = [-1.0, -1.0, 1.0]),
                        cplex.SparsePair(ind = [f"x_{i}", f"x_{j}",f"z_{i}{j}"], val = [-1.0, 1.0, 1.0]),
                        cplex.SparsePair(ind = [f"x_{i}", f"x_{j}",f"z_{i}{j}"], val = [1.0, -1.0, 1.0]),
                        cplex.SparsePair(ind = [f"x_{i}", f"x_{j}",f"z_{i}{j}"], val = [1.0, 1.0, 1.0])],
            senses = ["L", "G", "G", "L"],
            rhs = [0, 0, 0, 2],
            names = [f'c_{i}{j}0',f'c_{i}{j}1',f'c_{i}{j}2',f'c_{i}{j}3'])

    return prob, x_idx

def get_error(prob):
    """ Finds the variables for m in the problem """

    return get_name(prob, "violation")

def get_name(prob, keyword):
    """ Finds the variables for keyword in the problem """

    changed_name = ""
    for i in range(prob.linear_constraints.get_num()):
        name = prob.linear_constraints.get_names(i)
        if keyword in name:
            changed_name = name
            return changed_name

    if changed_name == "":
        print(f"Error: no {keyword}")
        return

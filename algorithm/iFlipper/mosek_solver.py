from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import sys
import time
import mosek

def MOSEK_Solver(label, m, w_sim, edge, ILP = False, verbose = False):
    """         
        Solves the label flipping optimization problem for a given total error limit m using MOSEK.
        If ILP is False, it solves the LP problem, otherwise solves the ILP problem.

        Args: 
            label: Labels of the data
            m: The total error limit
            w_sim: Similarity matrix
            edge: Indices of similar pairs
            ILP: Indicates the type of problem
            verbose: Prints log if True
            
        Return:
            flipped_label: Flipped labels for a given m
    """

    # Variables: concatnation of "[zi] - length n, [xi] - length n, [zij] - length m"
    n = label.shape[0]
    ne = len(edge)    
    with mosek.Env() as env:
        _ = 0.0
        with env.Task(0, 0) as task:
            if verbose:
                task.set_Stream(mosek.streamtype.log, streamprinter)
            else:
                task.set_Stream(mosek.streamtype.log, noprinter)

            # The variables will initially be fixed at zero (x=0).
            task.appendvars(2*n + ne)
            for j in range(n):
                task.putcj(j, 1.0)
                task.putvarbound(j, mosek.boundkey.ra, 0.0, 1.0)
            
            for j in range(n, 2*n):
                task.putcj(j, 0.0)
                task.putvarbound(j, mosek.boundkey.ra, 0.0, 1.0)
                if ILP:
                    task.putvartype(j, mosek.variabletype.type_int) # in case of ILP
            
            for j in range(2*n, 2*n + ne):
                task.putcj(j, 0.0)
                task.putvarbound(j, mosek.boundkey.ra, 0.0, 1.0)

            # The constraints will initially have no bounds.
            task.appendcons(2*n + 4*ne + 1)

            # Set the bounds on constraints.
                # blc[i] <= constraint_i <= buc[i]
            for k1 in range(n):
                ci = label[k1]
                # -yi' <= zi - yi <= yi'
                task.putconbound(2*k1+0, mosek.boundkey.ra, -ci, ci)
                task.putarow(2*k1+0, [k1, n+k1], [1.0, -1.0])

                # yi' <= zi + yi <= 2 - yi'
                task.putconbound(2*k1+1, mosek.boundkey.ra, ci, 2-ci)
                task.putarow(2*k1+1, [k1, n+k1], [1.0, 1.0])

            for k2, (i, j) in enumerate(edge):
                # + zij - yi - yj <= 0
                task.putconbound(2*n+4*k2+0, mosek.boundkey.up, _, 0.0)
                task.putarow(2*n+4*k2+0, [2*n + k2, n + i, n + j], [1.0, -1.0, -1.0])

                # + zij - yi + yj >= 0
                task.putconbound(2*n+4*k2+1, mosek.boundkey.lo, 0.0, _)
                task.putarow(2*n+4*k2+1, [2*n + k2, n + i, n + j], [1.0, -1.0, 1.0])

                # + zij + yi - yj >= 0
                task.putconbound(2*n+4*k2+2, mosek.boundkey.lo, 0.0, _)
                task.putarow(2*n+4*k2+2, [2*n + k2, n + i, n + j], [1.0, 1.0, -1.0])

                # + zij + yi + yj <= 2
                task.putconbound(2*n+4*k2+3, mosek.boundkey.up, _, 2.0)
                task.putarow(2*n+4*k2+3, [2*n + k2, n + i, n + j], [1.0, 1.0, 1.0])

            # sum sum Wij * zij <= m
            task.putconbound(2*n+4*ne, mosek.boundkey.up, _, m)
            
            task.putarow(2*n+4*ne, [i for i in range(2*n, 2*n+ne)], [w_sim[i][j] for (i, j) in edge])
            #task.putarow(2*n+4*ne, [i for i in range(2*n, 2*n + ne)], [1.0 for i in range(2*n, 2*n + ne)])

            # Input the objective sense (minimize/maximize)
            task.putobjsense(mosek.objsense.minimize)
            
            # Solve the problem
            task.optimize()

            # Print a summary containing information
            # about the solution for debugging purposes
            task.solutionsummary(mosek.streamtype.msg)

            # Get status information about the solution
            soln = [0.] * (2*n + ne)
            if ILP:
                task.getxx(mosek.soltype.itg, soln)
            else:
                task.getxx(mosek.soltype.bas, soln)

    return np.round(soln[n:2*n], decimals=4)

# Define a stream printer to grab output from MOSEK
def noprinter(text):
    pass

def streamprinter(text):
    sys.stdout.write(text)
    sys.stdout.flush()
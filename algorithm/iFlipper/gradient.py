from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import copy

import torch
from torch.nn.functional import relu
from torch import nn
import torch.optim as optim
from torch.distributions.relaxed_bernoulli import RelaxedBernoulli
try:
    from .utils import measure_error
except:
    from utils import measure_error

def Gradient(label, m, edge, w_edge, lam_high):
    """         
        Solves an unconstrained optimization problem via gradient descent.
        If the output does not satisfy the total error limit m, it increases lambda, which controls the trade-off between fairness and accuracy, up to lam_high.

        Args: 
            label: Labels of the data
            m: The total error limit
            edge: Indices of similar pairs
            w_edge: Similarity values of similar pairs
            lam_high: Upper bound of lambda
            
        Return:
            flipped_label: Flipped labels for a given m
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    n_optim_steps, learning_rate = 1000, 0.1
    lam_high, lam_low, temp = lam_high, 0, 0.1

    init_error = measure_error(label, edge, w_edge)
    best_flips, best_error = label.shape[0], init_error
    best_flips_fail, best_error_fail = label.shape[0], init_error

    while (lam_high-lam_low) > 0.01:
        lam = (lam_high + lam_low) / 2
        losses = []
        c = torch.from_numpy(label).to(device)
        w_edge_tensor = torch.from_numpy(w_edge).to(device)
        theta = torch.zeros_like(c).to(device)
        theta.requires_grad_()
        optimizer = optim.Adam([theta], lr=learning_rate)
        for ii in range(n_optim_steps):
            optimizer.zero_grad()
            loss = L_RB_Binary(c, theta, lam, temp, edge, w_edge_tensor)
            loss.backward()
            optimizer.step()
            losses.append(loss.cpu().detach().numpy())
            if ii % 100 == 99:
                optimizer.param_groups[0]['lr'] *= 0.5

        raw_label = torch.sigmoid(theta).cpu().detach().numpy()
        rounded_label = np.round(raw_label)

        total_error = measure_error(rounded_label, edge, w_edge)
        num_flips = np.sum(label != rounded_label)

        if total_error <= m: 
            if num_flips < best_flips:
                best_flips = num_flips
                best_error = total_error
                best_flipped_label = copy.deepcopy(rounded_label)
            lam_high = lam
        else:
            if total_error < best_error_fail:
                best_flips_fail = num_flips
                best_error_fail = total_error
                best_flipped_label_fail = copy.deepcopy(rounded_label)
            lam_low = lam

        if best_error <= m:
            flipped_label = copy.deepcopy(best_flipped_label)
        else:
            flipped_label = copy.deepcopy(best_flipped_label_fail)

    return flipped_label

def L_RB_Binary(c, theta, lam, temp, edge, w_edge):
    """         
        Computes (sum (yi-yi')**2) + lambda * (sum Wij(yi-yj)**2).
        Here we use the RelaxedBernoulli function to enforce the output label close to either 0 or 1.

        Args: 
            theta: Raw labels from the network
            c: Original labels
            lam: Hyperparameter lambda that controls the trade-off bewtween fairness and accuracy
            temp: Temperature parameter for RelaxedBernoulli
            edge: Indices of similar pairs
            w_edge: Similarity values of similar pairs

        Return:
            flipped_label: Flipped labels for a given m
    """

    dist = RelaxedBernoulli(temperature=temp, logits = theta)
    y = dist.rsample()
    return torch.sum((y-c)**2) + lam*torch.sum(w_edge * (y[edge[:,0]]-y[edge[:,1]])**2) 

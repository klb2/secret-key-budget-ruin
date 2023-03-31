from typing import Iterable
import numpy as np

def worst_case_outage_prob(mu: Iterable, var: Iterable, budget):
    if len(mu) != len(var):
        raise ValueError("The number of provided means and variances needs to be the same.")
    sum_var = np.sum(var)
    sum_mean = np.sum(mu)
    sum_bound = np.sqrt(sum_var+sum_mean**2)
    return np.minimum(sum_bound/budget, 1)

def worst_case_outage_prob_iid(mu: float, var: float, budget, n):
    sum_var = n*var
    sum_mean = n*mu
    sum_bound = np.sqrt(sum_var+sum_mean**2)
    return np.minimum(sum_bound/budget, 1)

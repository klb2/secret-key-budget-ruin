from typing import Union, Iterable

import numpy as np
from scipy import stats


def estimate_stop_times_samples(start_budget: float, acc_claims: Iterable):
    surplus = start_budget - acc_claims
    stop_times = np.argmax(surplus<=0, axis=1)
    stop_times = stop_times + 1
    num_timesteps = np.shape(acc_claims)[1]
    hist, bin_edges = np.histogram(stop_times,
                                   bins=np.arange(.5, num_timesteps+1.5, 1),
                                   density=True)
    return bin_edges+.5, hist


def mc_stopping_time(rv: Union[stats.rv_continuous, stats.rv_discrete],
                     num_samples: int = 100000,
                     num_timesteps: int = 100,
                     num_budgets: int = 200,
                     max_budget: float = 60.):
    samples_y = rv.rvs(size=(num_samples, num_timesteps-1))
    acc_claims = np.cumsum(samples_y, axis=1)

    budgets = np.linspace(0, max_budget, num_budgets)

    pdf = np.zeros((num_budgets, num_timesteps), dtype=float)
    for idx_budget, start_budget in enumerate(budgets):
        _tau, _pdf_tau = estimate_stop_times_samples(start_budget, acc_claims)
        _pdf_budget = np.append(0, _pdf_tau)
        pdf[idx_budget] = _pdf_budget
    cdf = np.cumsum(pdf, axis=1)
    return budgets, cdf.T

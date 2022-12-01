import logging
from typing import Iterable

import numpy as np
from scipy import stats
import matplotlib.pyplot as plt

from ide_stopping_time import ide_fft
from mc_stopping_time import mc_stopping_time

from util import export_results, find_closest_element_idx

LOGGER = logging.getLogger(__name__)


def main(save_b: Iterable[float]=[5., 10., 20.],
         num_timesteps: int = 100,
         num_budgets: int = 200, plot: bool = False, export: bool = False):
    #rv = stats.chi2(df=3)
    #rv = stats.expon
    #rv = stats.norm(loc=1)
    rv = stats.laplace_asymmetric(kappa=.5)
    max_budget = 4*max(save_b)

    budget_mc, cdf_mc = mc_stopping_time(rv, max_budget=max_budget,
                                         num_timesteps=num_timesteps)
    budget_th, cdf_th = ide_fft(rv.pdf, max_x=max_budget,
                                num_timesteps=num_timesteps,
                                num_points=2**16)
    
    for b in save_b:
        idx_b_mc = find_closest_element_idx(budget_mc, b)
        idx_b_th = find_closest_element_idx(budget_th, b)
        timeline = np.arange(num_timesteps)
        _cdf_mc_b = cdf_mc[:, idx_b_mc]
        _cdf_th_b = cdf_th[:, idx_b_th]
        if export:
            results = {
                       "time": timeline,
                       "mc": np.ravel(_cdf_mc_b),
                       "ide": np.ravel(_cdf_th_b),
                      }
            fname = f"ruin-cdf-time-b{b:.1f}.dat"
            export_results(results, fname)
        if plot:
            fig, axs = plt.subplots()
            axs.semilogy(timeline, _cdf_mc_b, label="Monte Carlo Simulation")
            axs.semilogy(timeline, _cdf_th_b, label="Numerical Solution")
            axs.set_xlabel("Time Step $t$")
            axs.set_ylabel("Outage Probability $\\varepsilon$")
            axs.set_title(f"Start Budget $b_0={b:.1f}$")


    #t = 5
    #plt.plot(budget_mc, cdf_mc[t, :], 'r')
    #plt.plot(budget_th, cdf_th[t, :], 'b--')
    #plt.xlabel("Start Budget")
    #plt.ylabel("Stopping Probability (CDF)")
    #plt.title(f"Timestep $t={t:d}$")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    ###
    # TODO: Add command line arguments that you need
    ###
    parser.add_argument("--plot", action="store_true")
    parser.add_argument("--export", action="store_true")
    parser.add_argument("-v", "--verbosity", action="count", default=0,
                        help="Increase output verbosity")
    args = vars(parser.parse_args())
    verb = args.pop("verbosity")
    logging.basicConfig(format="%(asctime)s - [%(levelname)8s]: %(message)s",
                        handlers=[
                            logging.FileHandler("main.log", encoding="utf-8"),
                            logging.StreamHandler()
                        ])
    loglevel = logging.WARNING - verb*10
    LOGGER.setLevel(loglevel)
    main(**args)
    plt.show()

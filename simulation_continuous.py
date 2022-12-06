import logging
from typing import Iterable
import gc

import numpy as np
from scipy import stats
import matplotlib.pyplot as plt

from ide_stopping_time import ide_fft
from mc_stopping_time import mc_stopping_time

from util import (export_results, find_closest_element_idx, db_to_linear,
                  capacity)

LOGGER = logging.getLogger(__name__)


def main(save_b: Iterable[float] = [5., 10., 20.],
         save_t: Iterable[int] = [1, 4, 10],
         snr_bob: float = 20,
         snr_eve: float = 10,
         num_timesteps: int = 50,
         num_budgets: int = 200,
         num_samples: int = int(1e6),
         skip_mc: bool = False,
         plot: bool = False, export: bool = False):
    LOGGER.info("Starting simulation...")
    LOGGER.debug(f"Number of MC samples: {num_samples:E}")
    LOGGER.debug(f"Number of timesteps: {num_timesteps:d}")
    LOGGER.debug(f"Number of budgets: {num_budgets:d}")


    snr_bob_lin = db_to_linear(snr_bob)
    snr_eve_lin = db_to_linear(snr_eve)
    rv_bob = stats.expon(scale=snr_bob_lin)
    rv_eve = stats.expon(scale=snr_eve_lin)
    LOGGER.info(f"Avg. SNR Bob: {snr_bob:.1f}dB")
    LOGGER.info(f"Avg. SNR Eve: {snr_eve:.1f}dB")
    
    LOGGER.debug("Determining the density of the claims")
    _num_samples_rv = int(1e5)
    samples_x = rv_bob.rvs(size=_num_samples_rv)
    samples_xt = rv_bob.rvs(size=_num_samples_rv)
    samples_y = rv_eve.rvs(size=_num_samples_rv)

    rate_sum = capacity(samples_x+samples_y)
    rate_eve = capacity(samples_y)
    rate_bob = capacity(samples_xt)
    samples_claims = -(rate_sum - rate_eve - rate_bob)

    LOGGER.debug(f"Average claim: {np.mean(samples_claims):.3f}bit")
    _hist = np.histogram(samples_claims, bins=300)
    rv = stats.rv_histogram(_hist)
    LOGGER.info("Performing Gaussian KDE...")
    rv_kde = stats.gaussian_kde(samples_claims)
    LOGGER.info("KDE finished.")
    #plt.hist(samples_claims, bins=200, density=True)
    #x = np.linspace(-10, 10, 1000)
    #plt.plot(x, rv_kde.pdf(x))
    #return
    LOGGER.debug("Density estimated.")

    del samples_x, samples_xt, samples_y, rate_bob, rate_eve, rate_sum, samples_claims
    gc.collect()

    max_budget = 1.5*max(save_b)

    if not skip_mc:
        LOGGER.info("Working on the Monte Carlo simulation...")
        budget_mc, cdf_mc = mc_stopping_time(rv, max_budget=max_budget,
                                            num_samples=num_samples,
                                            num_timesteps=num_timesteps)
    else:
        LOGGER.info("Skipping Monte Carlo simulation...")
        budget_mc = np.linspace(0, max_budget, num_budgets)
        cdf_mc = np.zeros((num_timesteps, num_budgets))
    LOGGER.info("Working on the numerical calculation...")
    budget_th, cdf_th = ide_fft(rv_kde.pdf, max_x=max_budget,
                                num_timesteps=num_timesteps,
                                num_points=2**13)
                                #num_points=2**20)
    LOGGER.info("Finished all calculations.")
    
    for b in save_b:
        idx_b_mc = find_closest_element_idx(budget_mc, b)
        idx_b_th = find_closest_element_idx(budget_th, b)
        timeline = np.arange(num_timesteps)
        _cdf_mc_b = np.ravel(cdf_mc[:, idx_b_mc])
        _cdf_th_b = np.ravel(cdf_th[:, idx_b_th])
        LOGGER.debug(f"b0={b:.1f}, t=4 and 10: {_cdf_th_b[[5,11]]}")
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
            axs.legend()
            axs.set_xlabel("Time Step $t$")
            axs.set_ylabel("Outage Probability $\\varepsilon$")
            axs.set_title(f"Start Budget $b_0={b:.1f}$")
            axs.set_xlim([0, num_timesteps])
            axs.set_ylim([1e-7, 1])

    if export:
        _components = (("mc", cdf_mc, budget_mc),
                       ("th", cdf_th, budget_th),
                      )
        for _name, _cdf, _budget in _components:
            results = {f"t{t:d}": _cdf[t] for t in save_t}
            results["budget"] = _budget
            fname = f"ruin-cdf-budget-{_name}.dat"
            export_results(results, fname)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("-b", "--save_b", type=float, nargs="+")
    parser.add_argument("-t", "--save_t", type=int, nargs="+")
    parser.add_argument("-n", "--num_samples", type=int, default=int(1e6),
                        help="Number of MC samples used for the simulation")
    parser.add_argument("--skip_mc", action="store_true")
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

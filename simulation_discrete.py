import logging
import itertools
from typing import Iterable

import numpy as np
from scipy import stats
import matplotlib.pyplot as plt

from ide_stopping_time import ide_fft, ide_discrete
from mc_stopping_time import mc_stopping_time

from util import (export_results, find_closest_element_idx, db_to_linear,
                  capacity)

LOGGER = logging.getLogger(__name__)


def joint_discrete(x, p_x):
    support = list(itertools.product(*x))
    joint_prob = list(itertools.product(*p_x))
    joint_prob = np.prod(joint_prob, axis=1)
    return support, joint_prob


def main(save_b: Iterable[float]=[5., 10., 20.],
         snr_bob: float = 20,
         snr_eve: float = 10,
         p_bob: float = .6,
         p_eve: float = .5,
         num_timesteps: int = 50,
         num_budgets: int = 200,
         num_samples: int = int(1e6),
         skip_mc: bool = False,
         plot: bool = False, export: bool = False):
    LOGGER.info("Starting simulation...")
    LOGGER.debug(f"Number of MC samples: {num_samples:E}")
    LOGGER.debug(f"Number of timesteps: {num_timesteps:d}")
    LOGGER.debug(f"Number of budgets: {num_budgets:d}")


    snr_bob_lin = db_to_linear([5, snr_bob])
    snr_eve_lin = db_to_linear([0, snr_eve])
    p_bob = [1-p_bob, p_bob]
    p_eve = [1-p_eve, p_eve]
    joint_supp, joint_prob = joint_discrete(x=(snr_bob_lin, snr_bob_lin, snr_eve_lin),
                                            p_x=(p_bob, p_bob, p_eve))
    joint_supp = np.array(joint_supp)
    claim_supp = -np.log2((1+joint_supp[:, 0]+joint_supp[:, 2])/((1+joint_supp[:, 1])*(1+joint_supp[:, 2])))
    claim_pdf = np.vstack((claim_supp, joint_prob)).T
    claim_pdf = claim_pdf[claim_pdf[:, 0].argsort()]
    #claim_cdf = np.cumsum(claim_pdf, axis=0)
    #claim_cdf[:, 0] = claim_pdf[:, 0]
    #claim_cdf = np.vstack(([-np.inf, 0], claim_cdf))
    LOGGER.debug(f"Joint PDF: {claim_pdf}")

    _error = 1e-3
    num_bins = int(np.ptp(claim_pdf[:, 0])/_error)
    _hist = np.histogram(claim_pdf[:, 0], weights=claim_pdf[:, 1], bins=num_bins)
    rv = stats.rv_histogram(_hist)
    LOGGER.debug("Density estimated.")

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
    #budget_th, cdf_th = ide_fft(rv.pdf, max_x=max_budget,
    budget_th, cdf_th = ide_discrete(claim_pdf,
                                     max_x=max_budget,
                                     num_timesteps=num_timesteps,
                                     num_points=2**18)
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


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("-b", "--save_b", type=float, nargs="+")
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

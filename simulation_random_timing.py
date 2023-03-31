import logging
from typing import Iterable
import gc

import numpy as np
from scipy import stats
from scipy import integrate
import matplotlib.pyplot as plt

from ide_stopping_time import ide_fft
from mc_stopping_time import mc_stopping_time
from bounds_stopping_time import worst_case_outage_prob_iid
from ultimate_ruin_prob import calculate_ultimate_ruin, calculate_adjustment_coefficient

from util import (export_results, find_closest_element_idx, db_to_linear,
                  capacity)

LOGGER = logging.getLogger(__name__)

def main(max_budget: float,
         save_b: Iterable[float],
         snr_bob: float = 20,
         snr_eve: float = 10,
         prob_tx: float = 0.5,
         num_timesteps: int = 150,
         num_budgets: int = 200,
         num_samples: int = int(1e6),
         skip_mc: bool = False,
         skip_adj: bool = False,
         plot: bool = False,
         export: bool = False):
    if max(save_b) > max_budget:
        raise ValueError("The maximum budget to save needs to be smaller than the maximum budget specified")
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
    _num_samples_rv = int(1e5) #1e5 # for KDE
    samples_x = rv_bob.rvs(size=_num_samples_rv)
    samples_y = rv_eve.rvs(size=_num_samples_rv)

    rate_sum = capacity(samples_x+samples_y)
    rate_eve = capacity(samples_y)
    rate_bob = capacity(samples_x)
    samples_skg = -(rate_sum-rate_eve)
    samples_tx = rate_bob
    samples = np.vstack((samples_skg, samples_tx))
    selected_block = np.random.choice(2, p=[1-prob_tx, prob_tx],
                                      size=_num_samples_rv)
    samples_net_claims = samples[selected_block, np.arange(_num_samples_rv)]

    mean_income = np.mean(-samples_skg)
    mean_tx = np.mean(samples_tx)
    crit_p = mean_income/(mean_income+mean_tx)
    LOGGER.debug(f"Average SKG: {mean_income:.3f} bit")
    LOGGER.debug(f"Average TX: {mean_tx:.3f} bit")
    LOGGER.info(f"Crit p: p={crit_p:.4f}")
    
    mean_claim = np.mean(samples_net_claims)
    var_claim = np.var(samples_net_claims)
    LOGGER.debug(f"Average claim: {mean_claim:.3f}bit")

    _hist = np.histogram(samples_net_claims, bins=300)
    rv = stats.rv_histogram(_hist)
    LOGGER.debug("Density estimated.")

    LOGGER.info("Performing Gaussian KDE...")
    rv_kde_claims = stats.gaussian_kde(samples_tx)
    rv_kde_income = stats.gaussian_kde(samples_skg)
    rv_kde = stats.gaussian_kde(samples_net_claims, .03)
    #__x = np.linspace(-10, 10, 1000)
    #plt.hist(samples_net_claims, bins=100, density=True)
    #plt.plot(__x, rv_kde(__x))
    #return
    LOGGER.info("KDE finished.")
    #print(np.mean(np.exp(-.5*samples_net_claims)))
    #__r = np.linspace(-.1, .1, 1000)
    #__g = [np.mean(np.exp(_r*samples_net_claims)) for _r in __r]
    #plt.semilogy(__r, __g)
    #plt.hlines(1, min(__r), max(__r))

    del samples_x, samples_y, rate_bob, rate_eve, rate_sum, samples_net_claims
    gc.collect()

    if not skip_mc:
        LOGGER.info("Working on the Monte Carlo simulation...")
        budget_mc, cdf_mc = mc_stopping_time(rv, max_budget=max_budget,
                                             num_samples=num_samples,
                                             num_timesteps=num_timesteps)
    else:
        LOGGER.info("Skipping Monte Carlo simulation...")
        budget_mc = np.linspace(0, max_budget, num_budgets)
        cdf_mc = np.zeros((num_timesteps, num_budgets))
    LOGGER.info("Finished the Monte Carlo simulation...")

    timeline = np.arange(num_timesteps)

    LOGGER.info("Calculating the probability of ultimate ruin...")
    budget_est, ult_ruin_prob_est = calculate_ultimate_ruin(rv_kde,
                                                            max_budget,
                                                            num_points=num_budgets)
    LOGGER.info("Finished calculating the probability of ultimate ruin...")

    if not skip_adj:
        LOGGER.info("Calculating the adjustment coefficient.")
        adj_coeff = calculate_adjustment_coefficient(rv.pdf, .5)
        #adj_coeff = calculate_adjustment_coefficient(rv_kde, .01)
        adj_coeff = -adj_coeff
    else:
        if prob_tx == 0.35:
            adj_coeff = -0.011058665791728244
        elif prob_tx == 0.1:
            adj_coeff = -0.28261788472119337
        else:
            raise NotImplementedError("The adjustment coefficient for this value of p is not stored.")
    LOGGER.info(f"Adjustment coefficient: r={adj_coeff:.4f}")
    ult_ruin_upper = np.exp(adj_coeff*budget_est)
    LOGGER.info("Finished all calculations.")

    if plot:
        fig, axs = plt.subplots()
        ult_ruin_prob_mc = cdf_mc[-1]
        axs.semilogy(budget_mc, ult_ruin_prob_mc, label="Monte Carlo")
        axs.semilogy(budget_est, ult_ruin_prob_est, label="Estimation")
        axs.semilogy(budget_est, ult_ruin_upper, label="Upper Bound")
    
    for b in save_b:
        idx_b_mc = find_closest_element_idx(budget_mc, b)
        _cdf_mc_b = np.ravel(cdf_mc[:, idx_b_mc])
        LOGGER.debug(f"Prob Ultimate Ruin (MC): b={b:.1f}, psi={_cdf_mc_b[-1]}")
        idx_b_est = find_closest_element_idx(budget_est, b)
        _prob_est = ult_ruin_prob_est[idx_b_est]
        _prob_up = ult_ruin_upper[idx_b_est]
        LOGGER.debug(f"Calculated Ultimate Ruin: b={b:.1f}, psi={_prob_est}")
        if plot:
            fig, axs = plt.subplots()
            axs.semilogy(timeline, _cdf_mc_b, label="Monte Carlo Simulation")
            axs.hlines(_prob_est, min(timeline), max(timeline), ls="--")
            axs.hlines(_prob_up, min(timeline), max(timeline), ls="--")
            axs.legend()
            axs.set_xlabel("Time Step $t$")
            axs.set_ylabel("Outage Probability $\\varepsilon$")
            axs.set_title(f"Start Budget $b_0={b:.1f}$")
            axs.set_xlim([0, num_timesteps])
            axs.set_ylim([1e-7, 1])
        if export:
            results = {
                       "time": timeline,
                       "mc": np.ravel(_cdf_mc_b),
                       "th": np.ones_like(timeline)*_prob_est,
                       "up": np.ones_like(timeline)*_prob_up,
                      }
            fname = f"ult-ruin-prob-time-b{b:.1f}-p{prob_tx:.2f}.dat"
            export_results(results, fname)

    if export:
        _components = (("mc", ult_ruin_prob_mc, budget_mc),
                       ("th", ult_ruin_prob_est, budget_est),
                       ("up", ult_ruin_upper, budget_est),
                      )
        results = {}
        for _name, _cdf, _budget in _components:
            results[_name] = _cdf
            results[f"budget{_name}"] = _budget
        fname = f"ult-ruin-prob-budget-p{prob_tx:.2f}.dat"
        export_results(results, fname)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("-b", "--save_b", type=float, nargs="+")
    parser.add_argument("-m", "--max_budget", type=float)
    parser.add_argument("-n", "--num_samples", type=int, default=int(1e6),
                        help="Number of MC samples used for the simulation")
    parser.add_argument("-p", "--prob_tx", type=float, default=0.5)
    parser.add_argument("--skip_mc", action="store_true")
    parser.add_argument("--skip_adj", action="store_true")
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

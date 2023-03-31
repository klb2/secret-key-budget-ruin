import logging

import numpy as np
from scipy import stats
import matplotlib.pyplot as plt

from util import export_results


LOGGER = logging.getLogger(__name__)

def main(b0=10, plot=False, export=False):
    net_claim = stats.norm(loc=-.7)
    income = stats.uniform(scale=2)

    current_budget = b0
    budget = [b0]

    LOGGER.info("Simulating Phase 1 (Net Claim)")
    while current_budget > 0:
        x = net_claim.rvs()
        current_budget += x
        current_budget = np.maximum(current_budget, 0)
        budget.append(current_budget)
    t_end = [len(budget)-1]
    LOGGER.info(f"End of Phase 1: {len(budget)-1:d}")

    LOGGER.info("Simulating Phase 2 (Recharging)")
    while current_budget < b0:
        z = income.rvs()
        current_budget += z
        budget.append(current_budget)
    t_end.append(len(budget)-1)
    LOGGER.info(f"End of Phase 2: {len(budget)-1:d}")

    LOGGER.info("Simulating Phase 3 (Net Claim)")
    while current_budget > 0:
        x = net_claim.rvs()
        current_budget += x
        current_budget = np.maximum(current_budget, 0)
        budget.append(current_budget)
    t_end.append(len(budget)-1)
    LOGGER.info(f"End of Phase 3: {len(budget)-1:d}")

    timeline = np.arange(len(budget))
    if plot:
        fig, axs = plt.subplots()
        axs.plot(timeline, budget, 'o-')
        axs.vlines(t_end, 0, b0)

    if export:
        results = {"t": timeline,
                   "b": budget}
        export_results(results, "samples_budget_illustration.dat")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
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

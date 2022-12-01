import logging

import matplotlib.pyplot as plt

from util import export_results


LOGGER = logging.getLogger(__name__)

def main(plot=False, export=False):
    ###
    # TODO: YOUR CODE HERE
    x = [1, 2, 3]
    y = [-5, 2, 1]
    results = {"x": x, "y": y}
    ###

    if plot:
        plt.plot(x, y)
    if export:
        LOGGER.info("Exporting results.")
        export_results(results, "results.dat")
    return results


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

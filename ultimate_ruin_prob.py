import logging
from typing import Callable

import numpy as np
import scipy
from scipy import stats
from scipy import integrate
from scipy import optimize
import matplotlib.pyplot as plt

LOGGER = logging.getLogger(__name__)

def calculate_ultimate_ruin(rv,
                            max_x: float,
                            num_points: int = 100):
    t, dt = np.linspace(0, max_x, num_points, retstep=True)
    t_mat = np.reshape(t, (-1, 1)) - t

    if True:
        inhomog = [1-rv.integrate_box_1d(-np.inf, _t) for _t in t]
        kernel_mat = [rv.pdf(row_t) for row_t in t_mat]
        kernel_mat = np.array(kernel_mat)
    else:
        inhomog = rv.sf(t)
        kernel_func = rv.pdf
        kernel_mat = kernel_func(t_mat)

    kernel_mat[:, 1:-1] *= 2
    matrix = dt/2 * kernel_mat
    
    outage_prob = scipy.linalg.solve(np.eye(num_points)-matrix, inhomog)
    return t, outage_prob


def calculate_adjustment_coefficient(pdf: Callable,
                                     #pdf_claim: Callable,
                                     #prob_tx: float,
                                     max_bracket=1,
                                     int_bounds=(-np.inf, np.inf)):
    def integrand(x, r):#, pdf):
        if pdf(x) == 0.: return 0.
        #return np.exp(r*x)*pdf(x)
        return np.exp(r*x + np.log(pdf(x)))
        #return np.exp(r*x + logpdf(x))

    def opt_func(r):
        if r == 0:
            return -np.finfo(float).eps
        _integral = integrate.quad(integrand, *int_bounds,
                                   args=(r,), limit=100)
        #_integral = integrate.quad(integrand, *int_bounds, args=(r,))
        #_expectation = (1-prob_tx)*_integral_income[0] + prob_tx*_integral_claim[1]
        _expectation = _integral[0]
        return _expectation-1

    root = optimize.root_scalar(opt_func,
                                bracket=(np.finfo(float).eps, max_bracket),
                                x0=max_bracket/2, x1=max_bracket/4)
    #root = optimize.minimize_scalar(opt_func, bracket=(min_bracket, 0))
    if root.converged:
        adj_coeff = root.root
    else:
        raise RuntimeError("Could not determine the adjustment coefficient.")
    return adj_coeff


if __name__ == "__main__":
    rv = stats.norm(loc=-.5)
    #calculate_ultimate_ruin(rv, num_points=200,
    #                        max_x=100)
    calculate_adjustment_coefficient(rv.pdf)
    plt.show()

from typing import Callable, Union, Iterable
import logging

import numpy as np

LOGGER = logging.getLogger(__name__)
LOGGER.setLevel(logging.DEBUG)

def ide_fft(kernel_func: Callable,
            max_x: float = 60,
            num_timesteps: int = 25,
            num_points: int = 2**16,
            kernel_cdf: bool = False):
    dx = 2*max_x/num_points
    xx = np.linspace(-2*max_x, 2*max_x-dx, 2*num_points)
    _idx_budget = np.where(np.logical_and(0 <= xx, xx <= max_x))[0]
    PAD = (abs(xx) <= max_x)
    if kernel_cdf:
        KERNEL = kernel_func(np.concatenate((xx, [2*max_x])))
        KERNEL = np.diff(KERNEL)
    else:
        KERNEL = kernel_func(xx)

    surv_prob_init = np.heaviside(xx, 0)
    results = [surv_prob_init[_idx_budget]]

    FKERNEL = np.fft.fft(KERNEL)
    surv_prob = surv_prob_init

    for t in range(1, num_timesteps):
        LOGGER.debug(f"Processing time step {t:d}/{num_timesteps+1:d}")
        surv_prob = np.heaviside(xx, 0)*surv_prob
        FN = np.fft.fft(surv_prob)
        surv_prob = dx * np.real(np.fft.fftshift(np.fft.ifft(FN*FKERNEL)))
        surv_prob = surv_prob*PAD
        results.append(surv_prob[_idx_budget])
    results = np.array(results)
    return xx[_idx_budget], 1.-results


def ide_discrete(pmf: Union[dict, np.array, Iterable[Iterable]],
                 max_x: float = 60,
                 num_timesteps: int = 25,
                 num_points: int = 2**16):
    if isinstance(pmf, dict):
        pmf = np.array(list(pmf.items()))
    pmf = np.array(pmf)
    pmf = pmf[pmf[:, 0].argsort()]
    max_x = np.maximum(max_x, num_timesteps*pmf[-1, 0])
    cdf = np.cumsum(pmf, axis=0)
    cdf[:, 0] = pmf[:, 0]
    cdf = np.vstack(([-np.inf, 0], cdf))
    LOGGER.debug(f"Claim PDF:\n{pmf}")
    LOGGER.debug(f"Claim CDF:\n{cdf}")
    if np.shape(pmf)[1] != 2:
        raise ValueError("Shape of the PMF needs to be Nx2")
    if not np.isclose(cdf[-1, 1], 1):
        raise ValueError("Probabilities do not sum up to one.")
    #budget0 = np.arange(num_points)*max_x/(num_points-1)
    #padding_width = num_points//2
    #budget0 = np.arange(-padding_width, num_points)*max_x/(num_points-1)
    budget0 = np.arange(-2*num_points, 2*num_points)*max_x/(num_points-1)
    _idx_budget = np.where(np.logical_and(0 <= budget0, budget0 <= max_x))[0]

    #surv_prob = np.zeros((num_timesteps, num_points))
    surv_prob = np.zeros((num_timesteps, len(budget0)))
    surv_prob[0] = np.heaviside(budget0, 0)
    surv_prob[1] = cdf[np.searchsorted(cdf[:, 0], budget0, 'right') - 1, 1]
    comb_u_t = budget0 - np.reshape(pmf[:, 0], (-1, 1))
    #idx_psi = np.searchsorted(budget0, comb_u_t)
    idx_psi = np.searchsorted(budget0, comb_u_t, 'right')-1
    idx_psi = np.clip(idx_psi, 0, len(budget0)-1)
    #idx_psi = np.clip(idx_psi, 0, num_points-1)
    assert np.shape(idx_psi) == (len(pmf), len(budget0))
    for timestep in range(2, num_timesteps):
        psi_prev = surv_prob[timestep-1][idx_psi]
        conv_matrix = psi_prev*np.reshape(pmf[:, 1], (-1, 1))
        surv_prob[timestep] = np.sum(conv_matrix, axis=0)
    budget0 = budget0[_idx_budget]
    surv_prob = surv_prob[:, _idx_budget]
    return budget0, 1.-surv_prob


if __name__ == "__main__":
    #pmf = [[5, .2],
    #       [4.2, .3],
    #       [2.1, .2],
    #       [.1, .05],
    #       [-1, .25]]
    pmf = [[-3.61505213, 0.12],
           [-1.27761104, 0.12],
           [0.68936547, 0.08],
           [0.98578614, 0.18],
           [1.69282343, 0.08],
           [3.32322724, 0.18],
           [5.29020374, 0.12],
           [6.2936617, 0.12]]
    max_x = 21
    num_points = 1000
    budget0, results = ide_discrete(pmf, max_x, 20, num_points)
    import matplotlib.pyplot as plt
    CMAP = plt.get_cmap("viridis")
    for idx, row in enumerate(results):
        plt.plot(budget0, row,
        #plt.plot(np.diff(row),
                 c=CMAP(idx/len(results)), label=f"t={idx:d}")
        plt.xlabel("Initial Budget $b_0$")
        plt.ylabel("Outage Probability $\\varepsilon$")
    plt.legend()
    plt.show()
    #print(results)

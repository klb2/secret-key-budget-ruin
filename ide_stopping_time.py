from typing import Callable

import numpy as np


def ide_fft(kernel_func: Callable,
            max_x: float = 60,
            num_timesteps: int = 25,
            num_points: int = 2**16):
    dx = 2*max_x/num_points
    xx = np.linspace(-2*max_x, 2*max_x-dx, 2*num_points)
    _idx_budget = np.where(np.logical_and(0 <= xx, xx <= max_x))[0]
    PAD = (abs(xx) <= max_x)
    KERNEL = kernel_func(xx)

    surv_prob_init = np.heaviside(xx, 0)
    results = [surv_prob_init[_idx_budget]]

    FKERNEL = np.fft.fft(KERNEL)
    surv_prob = surv_prob_init

    for t in range(1, num_timesteps):
        surv_prob = np.heaviside(xx, 0)*surv_prob
        FN = np.fft.fft(surv_prob)
        surv_prob = dx * np.real(np.fft.fftshift(np.fft.ifft(FN*FKERNEL)))
        surv_prob = surv_prob*PAD
        results.append(surv_prob[_idx_budget])
    results = np.array(results)
    return xx[_idx_budget], 1.-results

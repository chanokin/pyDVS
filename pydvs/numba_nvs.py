from numba import jit
import numpy as np

@jit(nopython=True)
def nvs_step(
        frame: np.ndarray, reference: np.ndarray, thresholds: np.ndarray,
        reference_leak: np.float,
        threshold_base: np.float,
        threshold_mult_incr: np.float, threshold_leak: np.float):
    active = np.zeros_like(frame, dtype='uint8')
    diff = np.zeros_like(frame)
    diff[:] = frame - reference
    active[:] = np.abs(diff) > thresholds
    diff *= active

    # threshold is basically a leaky integrator
    thresholds[:] = (
            threshold_base + (thresholds - threshold_base) * threshold_leak +
            (thresholds - threshold_base) * threshold_mult_incr * active
    )

    # reference is also a leaky integrator
    # todo: since outputs are spikes, how do I discretize increments?
    reference[:] = (
            reference * reference_leak + diff
    )

    # diff are now the 'spikes'
    return diff, reference, thresholds


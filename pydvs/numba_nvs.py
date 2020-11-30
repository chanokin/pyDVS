from numba import jit, float32, uint8
import numpy as np


@jit(parallel=True, nopython=True)
def thresholded_difference(
        frame: np.ndarray, reference: np.ndarray, thresholds: np.ndarray,
        spikes: np.ndarray):
    active = np.zeros_like(frame, dtype=uint8)

    spikes[:] = frame - reference
    active[:] = (np.abs(spikes) >= thresholds)
    spikes *= active
    # todo: since outputs are spikes, how do I discretize increments?
    # is this the right way?
    spikes[:] = (spikes // thresholds) * thresholds

    return active


@jit(nopython=True)
def nvs_0(frame: np.ndarray, reference: np.ndarray, thresholds: np.ndarray,
          spikes: np.ndarray):
    active = np.zeros_like(frame, dtype=uint8)

    active[:] = thresholded_difference(frame, reference, thresholds, spikes)
    reference[:] += spikes


@jit(nopython=True)
def nvs_leaky(frame: np.ndarray, reference: np.ndarray, thresholds: np.ndarray,
              spikes: np.ndarray, reference_leak: np.float):
    active = np.zeros_like(frame, dtype=uint8)

    active[:] = thresholded_difference(frame, reference, thresholds, spikes)

    # reference is a leaky integrator
    reference[:] = reference * reference_leak + spikes


@jit(nopython=True)
def nvs_leaky_adaptive(frame: np.ndarray, reference: np.ndarray,
                       thresholds: np.ndarray, spikes: np.ndarray,
                       reference_leak: np.float,
                       threshold_base: np.float,
                       threshold_mult_incr: np.float, threshold_leak: np.float):
    active = np.zeros_like(frame, dtype=uint8)

    active[:] = thresholded_difference(frame, reference, thresholds, spikes)

    # thresholds as leaky integrators
    thresholds[:] = (
            threshold_base + (thresholds - threshold_base) * threshold_leak +
            (thresholds - threshold_base) * threshold_mult_incr * active
    )

    # reference is also a leaky integrator
    reference[:] = (
            reference * reference_leak + spikes
    )



@jit(nopython=True)
def nvs_noisy_leaky_adaptive(frame: np.ndarray, reference: np.ndarray,
                             thresholds: np.ndarray, spikes: np.ndarray,
                             reference_leak: np.float,
                             threshold_base: np.float,
                             threshold_mult_incr: np.float, threshold_leak: np.float,
                             reference_leak_probability: np.float):
    active = np.zeros_like(frame, dtype=uint8)
    leak_active = np.ones_like(frame, dtype=float32)

    active[:] = thresholded_difference(frame, reference, thresholds, spikes)

    # thresholds as leaky integrators
    thresholds[:] = (
            threshold_base + (thresholds - threshold_base) * threshold_leak +
            (thresholds - threshold_base) * threshold_mult_incr * active
    )

    # reference is also a leaky integrator
    leak_active += (np.random.uniform(0., 1., size=frame.shape) <=
                    reference_leak_probability) * (reference_leak - 1)

    reference[:] = (
            reference * leak_active + spikes
    )


@jit(nopython=True)
def nvs_bi_noisy_leaky_adaptive(
        frame: np.ndarray, reference: np.ndarray, thresholds: np.ndarray,
        spikes: np.ndarray,
        reference_leak: np.float, threshold_base: np.float,
        threshold_mult_incr: np.float, threshold_leak: np.float,
        reference_leak_probability: np.float,
        reference_off: np.ndarray, thresholds_off: np.ndarray, target_off: np.float):
    active = np.zeros_like(frame, dtype=uint8)
    spikes_off = np.zeros_like(frame, dtype=float32)
    active_off = np.zeros_like(frame, dtype=float32)
    leak_active = np.ones_like(frame, dtype=float32)

    active[:] = thresholded_difference(frame, reference, thresholds, spikes)
    active_off[:] = thresholded_difference(
                            frame, reference_off, thresholds_off, spikes_off)
    # thresholds as leaky integrators
    thresholds[:] = (
        threshold_base + (thresholds - threshold_base) * threshold_leak +
        (thresholds - threshold_base) * threshold_mult_incr * active
    )

    # reference is also a leaky integrator
    leak_active += (np.random.uniform(0., 1., size=frame.shape) <=
                    reference_leak_probability) * (reference_leak - 1)
    reference[:] = (
            reference * leak_active + spikes
    )

    thresholds_off[:] = (
        threshold_base + (thresholds_off - threshold_base) * threshold_leak +
        (thresholds_off - threshold_base) * threshold_mult_incr * active_off
    )

    leak_active[:] = 1.0
    leak_active += (np.random.uniform(0., 1., size=frame.shape) <=
                    reference_leak_probability) * (reference_leak - 1)
    reference_off[:] = (
            target_off + (reference - target_off) * leak_active + spikes_off
    )

    # think: what's faster?
    # for row in range(frame.shape[0]):
    #     for col in range(frame.shape[1]):
    #         if spikes[row, col] < 0:
    #             spikes[row, col] = 0
    #
    #         if spikes_off[row, col] > 0:
    #             spikes_off[row, col] = 0
    #
    #         if (-spikes_off[row, col]) > spikes[row, col]:
    #             spikes[row, col] = spikes_off[row, col]


    # spikes_off *= spikes_off < 0
    # spikes *= spikes > 0
    # spikes += spikes_off * (np.abs(spikes_off) > spikes)

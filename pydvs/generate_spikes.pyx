from cython.parallel import prange
import array
from cpython cimport array
import time
import cv2
import numpy as np
cimport numpy as np

cimport cython

from pydvs.cdefines cimport *
from pydvs.pdefines import *

from pydvs.math_utils cimport *
from pydvs.math_utils_np import *

@cython.boundscheck(False)  # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
cdef thresholded_difference(np.ndarray[DTYPE_t, ndim=2] curr_frame,
                            np.ndarray[DTYPE_t, ndim=2] ref_frame,
                            np.ndarray[DTYPE_t, ndim=2] threshold):
    """
    :param curr_frame: Latest image captured by the camera
    :param ref_frame:  Saves value when the pixel was marked as "spiking"
    :param threshold:  How big the difference between current and reference
                       frames needs to be to mark a pixel as spiking. Adjusted
                       dynamically.

    :return diff:     Signed difference between current and reference frames
    :return abs_diff: Absolute value of the difference
    :return spikes:   Signed pixels marked as spiking (-1 means change from higher
                      to lower brightness value, 1 means change from lower to
                      higher brightness value.)
    """
    cdef np.ndarray[DTYPE_t, ndim= 2] diff, abs_diff, spikes

    diff = curr_frame - ref_frame
    abs_diff = np.abs(diff)

    spikes = (abs_diff > threshold).astype(DTYPE)
    abs_diff = (abs_diff*spikes)

    spikes[:] = np.round( (diff * spikes) / threshold )

    return diff, abs_diff, spikes


@cython.boundscheck(False)  # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
cdef update_threshold(np.ndarray[DTYPE_t, ndim=2] threshold,
                      np.ndarray[DTYPE_t, ndim=2] spikes,
                      DTYPE_t mult_up, DTYPE_t mult_down, DTYPE_t base_level):
    """
    Should be called after update_reference!
    :param threshold: current thresholds value
    :param spikes: how many thresholds did pixel value (in)decreased
    :param mult_up: multiplicative update for thresholds, value > 1 
    :param mult_down: how fast should the thresholds return to base_level, 0 < val < 1
    :param base_level: base level for thresholds
    """
    if mult_up <= 1.0 or mult_down >= 1.0:
        return threshold
    
    cdef np.ndarray[DTYPE_U8_t, ndim=2] mask
    mask = (spikes == 0) # didn't spike
    threshold[~mask] *= mult_up # inv didn't spike
    threshold[mask] = base_level + (threshold[mask] - base_level) * mult_down

    return threshold

@cython.boundscheck(False)  # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
cdef update_reference(np.ndarray[DTYPE_t, ndim=2] reference,
                      np.ndarray[DTYPE_t, ndim=2] spikes,
                      np.ndarray[DTYPE_t, ndim=2] threshold,
                      DTYPE_t mult_down, DTYPE_t base_level, 
                      DTYPE_t min_level, DTYPE_t max_level):
    """
    Should be called before updating the thresholds!
    :param reference: current reference values
    :param spikes: how many thresholds did pixel value (in)decreased
    :param threshold: current thresholds value (before updating them!)
    :param mult_down: how fast should the reference return to base_level, 0 < val < 1
    :param base_level: base level for references
    :param min_level: lower bound of the value range for references (min pixel value)
    :param max_level: upper bound of the value range for references (max pixel value)
    """

    reference += (spikes * threshold)
    if mult_down >= 1.0:
        return reference

    cdef np.ndarray[DTYPE_U8_t, ndim=2] mask
    mask = (spikes == 0)
    reference[mask] = np.clip(base_level + (reference[mask] - base_level) * mult_down, 
                            min_level, max_level)
    return reference

@cython.boundscheck(False)  # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
cdef get_output_spikes(np.ndarray[DTYPE_t, ndim=2] abs_on, 
                       np.ndarray[DTYPE_t, ndim=2] abs_off,
                       np.ndarray[DTYPE_t, ndim=2] spikes_on, 
                       np.ndarray[DTYPE_t, ndim=2] spikes_off):
    """
    """
    cdef np.ndarray[DTYPE_t, ndim=2] off_mask
    cdef np.ndarray[DTYPE_t, ndim=2] on_mask
    cdef np.ndarray[DTYPE_t, ndim=2] on_out
    cdef np.ndarray[DTYPE_t, ndim=2] off_out

    off_mask = spikes_off < 0
    on_mask = spikes_on > 0
    
    on_out = spikes_on * ((on_mask * abs_on) > abs_off)
    off_out = -spikes_off * ((off_mask * abs_off) > abs_on)

    return on_out, off_out

@cython.boundscheck(False)  # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
cdef local_inhibition(np.ndarray[DTYPE_t, ndim=2] spikes,
                     np.ndarray[DTYPE_t, ndim=2] abs_diff,
                     np.ndarray[DTYPE_t, ndim=2] inh_coords,
                     DTYPE_IDX_t width,
                     DTYPE_IDX_t height,
                     DTYPE_IDX_t inh_width):
    """ Searches for the largest change in an area and removes spiking mark
        from the other pixels.
        :param spikes:      Marks of pixels
        :param abs_diff:    Absolute value of the brightness change
        :param inh_coords:  Precomputed coordinate transformation
        :param width:       Image width
        :param height:      Image height
        :param inh_width:   Inhibition area width (square area)
        :returns spikes:    Locally inhibited spikes
    """
    cdef DTYPE_IDX_t u, v, new_w, new_h, ratio

# ~   cdef np.ndarray[DTYPE_t, ndim=2] reshaped
    cdef DTYPE_t[:, :] reshaped
    cdef np.ndarray[DTYPE_t, ndim= 1] max_vals
    cdef np.ndarray[Py_ssize_t, ndim= 2] max_indices
    cdef np.ndarray[Py_ssize_t, ndim= 1] max_local_indices

    new_w = inh_width*inh_width
    new_h = (height*width)//new_w
    ratio = width//inh_width

    # (w*h/inh_w^2, inh_w^2) so we can efficiently find local max
    reshaped = abs_diff[inh_coords[:, ROWS],
                        inh_coords[:, COLS]].reshape((new_h, new_w))
    max_local_indices = np.zeros(new_h, dtype=DTYPE_IDX)
    # allow parallel compute of max indices
    for v in prange(new_h, nogil=True):
        max_local_indices[v] = argmax(reshaped[v], new_w)

    max_indices = np.array([transform_coords(u, max_local_indices[u], inh_width, ratio)
                            for u in range(new_h)], dtype=DTYPE_IDX)

    max_vals = spikes[max_indices[:, ROWS], max_indices[:, COLS]]
    spikes[:] = 0
    spikes[max_indices[:, ROWS], max_indices[:, COLS]] = max_vals

    return spikes



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

cdef thresholded_difference(np.ndarray[DTYPE_t, ndim=2] curr_frame,
                            DTYPE_t[:, :] ref_frame,
                            DTYPE_t[:, :] threshold)

cdef update_threshold(DTYPE_t[:, :] threshold,
                      DTYPE_t[:, :] spikes,
                      DTYPE_t mult_up, DTYPE_t mult_down, DTYPE_t base_level)

cdef update_reference(DTYPE_t[:, :] reference,
                      DTYPE_t[:, :] spikes,
                      DTYPE_t[:, :] threshold,
                      DTYPE_t mult_down, DTYPE_t base_level, 
                      DTYPE_t min_level, DTYPE_t max_level)

cdef get_output_spikes(np.ndarray[DTYPE_t, ndim=2] abs_on, 
                       np.ndarray[DTYPE_t, ndim=2] abs_off,
                       DTYPE_t[:, :] spikes_on, 
                       DTYPE_t[:, :] spikes_off)

cdef local_inhibition(DTYPE_t[:, :] spikes,
                     np.ndarray[DTYPE_t, ndim=2] abs_diff,
                     np.ndarray[DTYPE_t, ndim=2] inh_coords,
                     DTYPE_IDX_t width,
                     DTYPE_IDX_t height,
                     DTYPE_IDX_t inh_width)
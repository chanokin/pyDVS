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

cpdef render_frame(np.ndarray[DTYPE_t, ndim=2] spikes,
                 np.ndarray[DTYPE_t, ndim=2] curr_frame,
                 DTYPE_t width,
                 DTYPE_t height,
                 DTYPE_U8_t polarity)

cpdef render_comparison(np.ndarray[DTYPE_t, ndim=2] curr_frame,
                      np.ndarray[DTYPE_t, ndim=2] ref_frame,
                      np.ndarray[DTYPE_t, ndim=2] lap_curr,
                      np.ndarray[DTYPE_t, ndim=2] lap_ref,
                      np.ndarray[DTYPE_U8_t, ndim=3] spikes_frame,
                      DTYPE_IDX_t width, DTYPE_IDX_t height)
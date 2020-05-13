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

@cython.boundscheck(False)  # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
cpdef render_frame(np.ndarray[DTYPE_t, ndim=2] spikes,
                 np.ndarray[DTYPE_t, ndim=2] curr_frame,
                 DTYPE_IDX_t width,
                 DTYPE_IDX_t height):
    """
      Overlaps the generated spikes onto the latest image from the video
      source. Red means a negative change in brightness, Green a positive one.

      :param spikes:     Pixels marked as spiking
      :param curr_frame: Latest image from the video source
      :param width:      Image width
      :param height:     Image height
      :returns spikes_frame: Combined spikes/image information in a color image
    """
    cdef np.ndarray[DTYPE_U8_t, ndim= 3] spikes_frame = \
                                np.zeros([height, width, 3], dtype=DTYPE_U8)
    cdef np.ndarray[DTYPE_IDX_t, ndim= 1] rows, cols

    spikes_frame[:, :, 0] = curr_frame
    spikes_frame[:, :, 1] = curr_frame
    spikes_frame[:, :, 2] = curr_frame

    rows, cols = np.where(spikes > 0)
    spikes_frame[rows, cols, :] = [0, 200, 0]

    rows, cols = np.where(spikes < 0)
    spikes_frame[rows, cols, :] = [0, 0, 200]

    return spikes_frame


@cython.boundscheck(False)  # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
cpdef render_comparison(np.ndarray[DTYPE_t, ndim=2] curr_frame,
                      np.ndarray[DTYPE_t, ndim=2] ref_frame,
                      np.ndarray[DTYPE_t, ndim=2] lap_curr,
                      np.ndarray[DTYPE_t, ndim=2] lap_ref,
                      np.ndarray[DTYPE_U8_t, ndim=3] spikes_frame,
                      DTYPE_IDX_t width, DTYPE_IDX_t height):
    """
      Compose a comparison of visual features
      -----------------------------------------------
      | current   |  reference  |   abs difference  |
      -----------------------------------------------
      | edges in  |  edges in   |   abs difference  |
      | current   |  reference  |   of edges        |
      -----------------------------------------------

    """
# ~   cdef np.ndarray[DTYPE_U8_t, ndim=3] out = np.zeros([2*height, 4*width, 3], dtype=DTYPE_U8)
    cdef np.ndarray[DTYPE_U8_t, ndim= 3] out = np.zeros([height, 4*width, 3], dtype=DTYPE_U8)

    out[:height, 0:width, 0] = curr_frame
    out[:height, 0:width, 1] = curr_frame
    out[:height, 0:width, 2] = curr_frame
    out[:height, width:2*width, 0] = ref_frame
    out[:height, width:2*width, 1] = ref_frame
    out[:height, width:2*width, 2] = ref_frame
# ~   out[:, 2*width:3*width, 0] = (np.abs(curr_frame - ref_frame) > 0)*255
# ~   out[:, 2*width:3*width, 1] = (np.abs(curr_frame - ref_frame) > 0)*255
# ~   out[:, 2*width:3*width, 2] = (np.abs(curr_frame - ref_frame) > 0)*255
    out[:height, 2*width:3*width, 0] = np.abs(curr_frame - ref_frame)
    out[:height, 2*width:3*width, 1] = np.abs(curr_frame - ref_frame)
    out[:height, 2*width:3*width, 2] = np.abs(curr_frame - ref_frame)
    out[:height, 3*width:, :] = spikes_frame

# ~   out[height:, 0:width, 0] = lap_curr
# ~   out[height:, 0:width, 1] = lap_curr
# ~   out[height:, 0:width, 2] = lap_curr
# ~   out[height:, width:2*width, 0] = lap_ref
# ~   out[height:, width:2*width, 1] = lap_ref
# ~   out[height:, width:2*width, 2] = lap_ref
# ~   out[height:, 2*width:3*width, 0] = np.abs(lap_curr - lap_ref)
# ~   out[height:, 2*width:3*width, 1] = np.abs(lap_curr - lap_ref)
# ~   out[height:, 2*width:3*width, 2] = np.abs(lap_curr - lap_ref)

    return out

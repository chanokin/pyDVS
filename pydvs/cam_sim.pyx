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


cpdef mask_image(np.ndarray[DTYPE_t, ndim=2] original,
               np.ndarray[DTYPE_FLOAT_t, ndim=2] mask):

    return DTYPE(original*mask)


cpdef traverse_image(np.ndarray[DTYPE_t, ndim=2] original,
                   DTYPE_t frame_number, DTYPE_FLOAT_t speed, DTYPE_t bg_gray):
    """
      Animate input image (original) at a certain rate (fps)
      :param original:     As loaded from memory
      :param fps:          Frames per second of the animation
      :param frame_number: How many frames  have passed since the first
                           appearence of the original image
      :param speed:        How many pixels per frame should the image move
      :returns moved:      Translated image
    """
    cdef np.ndarray[DTYPE_t, ndim= 2] moved = bg_gray*np.ones_like(original, dtype=DTYPE)
    cdef DTYPE_IDX_t width = len(original[0])
    cdef DTYPE_IDX_t n = DTYPE(frame_number*speed)

    if n == 0:
        n = 1

    if n <= width:
        moved[:, :n] = original[:, -n:]
    elif n < 2*width:
        n = 2*width - n + 1
        moved[:, -n:] = original[:, :n]

    return moved


cpdef fade_image(np.ndarray[DTYPE_t, ndim=2] original,
               DTYPE_IDX_t frame_number, DTYPE_IDX_t half_frame,
               DTYPE_t bg_gray):
    """
      Animate input image (original) at a certain rate (fps)
      :param original:     As loaded from memory
      :param fps:          Frames per second of the animation
      :param frame_number: How many frames  have passed since the first
                           appearence of the original image
      :param half_frame:   Frame at half the image's on time
      :returns moved:      Translated image
    """
    cdef np.ndarray[DTYPE_t, ndim= 2] moved = bg_gray*np.ones_like(original, dtype=DTYPE)

    cdef DTYPE_t alpha = 0.0
    if frame_number < half_frame - 1:
        alpha = DTYPE(frame_number) / DTYPE(half_frame)
    elif frame_number > half_frame + 1:
        alpha = DTYPE((half_frame*2) - frame_number) / DTYPE(half_frame)
    else:
        alpha = 1.0

    moved[:] = DTYPE(original*alpha)

    return moved


cdef move_image(np.ndarray[DTYPE_t, ndim=2] original,
                DTYPE_t delta_x,
                DTYPE_t delta_y,
                DTYPE_t bg_gray):
    cdef np.ndarray[DTYPE_t, ndim= 2] moved = bg_gray*np.ones_like(original, dtype=DTYPE)
    cdef DTYPE_t new_x0, new_x1, old_x0, old_x1
    cdef DTYPE_t new_y0, new_y1, old_y0, old_y1
    cdef DTYPE_t width = len(original[0])
    cdef DTYPE_t height = len(original)

    if delta_x < 0:
        new_x0 = 0
        new_x1 = delta_x
        old_x0 = abs(delta_x)
        old_x1 = width
    elif delta_x > 0:
        new_x0 = delta_x
        new_x1 = width
        old_x0 = 0
        old_x1 = -delta_x
    else:
        new_x0 = 0
        new_x1 = width
        old_x0 = 0
        old_x1 = width

    if delta_y > 0:
        new_y0 = 0
        new_y1 = -delta_y
        old_y0 = delta_y
        old_y1 = height
    elif delta_y < 0:
        new_y0 = abs(delta_y)
        new_y1 = height
        old_y0 = 0
        old_y1 = delta_y
    else:
        new_y0 = 0
        new_y1 = height
        old_y0 = 0
        old_y1 = height

    moved[new_y0:new_y1, new_x0:new_x1] = original[old_y0:old_y1, old_x0:old_x1]
    return moved


cpdef usaccade_image(np.ndarray[DTYPE_t, ndim=2] original,
                   DTYPE_t frame_number,
                   DTYPE_t frames_per_usaccade,
                   DTYPE_t max_delta,
                   DTYPE_t center_x,
                   DTYPE_t center_y,
                   DTYPE_t bg_gray):
    """
      Animate input image (original) at a certain rate (fps)
      :param original:     Previous image
      :param frame_number: How many frames  have passed since the first
                           appearence of the original image
      :param frames_per_usaccade: How many frames does previous image stay
                                  the same.
      :param max_delta:    How many pixels per frame should the image move
      :returns moved:      Translated image
    """
    if frame_number % frames_per_usaccade != 0:
        return move_image(original, center_x, center_y, bg_gray), center_x, center_y

    np.random.seed(seed=np.uint32(time.time()*10000000000))
    center_x += np.random.randint(-max_delta, max_delta+1)
    center_y += np.random.randint(-max_delta, max_delta+1)

    return move_image(original, center_x, center_y, bg_gray), center_x, center_y


cpdef attention_image(np.ndarray[DTYPE_t, ndim=2] original,
                    np.ndarray[DTYPE_t, ndim=2] previous,
                    np.ndarray[DTYPE_t, ndim=2] reference,
                    DTYPE_t frame_number,
                    DTYPE_t frames_per_usaccade,
                    DTYPE_t frame_saccade,
                    DTYPE_t max_delta,
                    DTYPE_t center_x,
                    DTYPE_t center_y,
                    DTYPE_t bg_gray):
    """
      Animate input image (original) at a certain rate (fps)
      :param previous:      Previous image
      :param original:      Original image, as read from memory
      :param frame_number:  Number of frames that have passed since image first appeared
      :param frame_saccade: When frame_number is a multiple of this, go to random highest
                            change position (saccade).
      :param max_delta:     How many pixels per frame should the image move (microsaccade)
      :returns moved:   Translated image
    """

    # not the right frame, do micro-saccade
    if frame_number % frame_saccade != 0:
        return usaccade_image(original, frame_number, frames_per_usaccade,
                              max_delta, center_x, center_y, bg_gray)

    # simulate saccade
    cdef np.ndarray[DTYPE_t, ndim= 2] moved = bg_gray*np.ones_like(original, dtype=DTYPE)
    cdef int new_w = len(original[0])//2
    cdef np.ndarray[DTYPE_t, ndim = 2] tiny_prev = cv2.resize(previous, (new_w, new_w),
                                                             interpolation=CV_INTER_AREA).astype(DTYPE)

    cdef np.ndarray[DTYPE_t, ndim = 2] tiny_ref  = cv2.resize(reference, (new_w, new_w),
                                                             interpolation=CV_INTER_AREA).astype(DTYPE)
    cdef np.ndarray[DTYPE_t, ndim= 2] diff = np.abs(tiny_prev - tiny_ref).astype(DTYPE)

    cdef np.ndarray[DTYPE_IDX_t, ndim= 1] top_n = np.argsort(diff.reshape(new_w*new_w))\
        .astype(DTYPE_IDX)[-5:]

    cdef int max_idx = top_n[np.random.randint(5)]
    cdef int row = (max_idx//new_w)*2
    cdef int col = (max_idx % new_w)*2

    center_x = new_w - col
    center_y = new_w - row
    if center_x > new_w or center_x < -new_w or\
       center_y > new_w or center_y < -new_w:
        center_x = 0
        center_y = 0

    return move_image(original, center_x, center_y, bg_gray), center_x, center_y

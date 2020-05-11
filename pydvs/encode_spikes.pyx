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

# from pydvs.math_utils cimport *
# from pydvs.math_utils_np cimport *

@cython.boundscheck(False)  # turn off bounds-checking for entire function
def split_spikes(np.ndarray[DTYPE_t, ndim=2] spikes,
                 np.ndarray[DTYPE_t, ndim=2] abs_diff,
                 DTYPE_U8_t polarity):
    """
      Divide spikes into positive and negative, both refering to change in
      brightness.
      :param spikes:    Pixels marked as spiking
      :param abs_diff:  The absolute value of the difference of current and the
                        reference frame.
      :param polarity:  Which spikes to report to the user.
      :returns negative: Pixels that spiked with a negative change in brightness
      :returns positive: Pixels that spiked with a positive change in brightness
      :returns global_max: Value of the largest change in brightness

      output format is [[row0, col0, val0], ... [rowN, colN, valN]]
    """
    cdef np.ndarray[DTYPE_IDX_t, ndim= 1] neg_rows, neg_cols, \
        pos_rows, pos_cols
    cdef np.ndarray[DTYPE_t, ndim= 1]    neg_vals, pos_vals
    cdef DTYPE_t global_max = 0

    if polarity == RECTIFIED_POLARITY:
        # print("RECTIFIED_POLARITY")
        pos_rows, pos_cols = np.where(spikes != 0)
        pos_vals = abs_diff[pos_rows, pos_cols]
        neg_rows = neg_cols = np.array([], dtype=DTYPE_IDX)
        neg_vals = np.array([], dtype=DTYPE)
        if len(pos_vals) > 0:
            global_max = pos_vals.max()

    elif polarity == UP_POLARITY:
        pos_rows, pos_cols = np.where(spikes > 0)
        pos_vals = abs_diff[pos_rows, pos_cols]
        neg_rows = neg_cols = np.array([], dtype=DTYPE_IDX)
        neg_vals = np.array([], dtype=DTYPE)

        if len(pos_vals) > 0:
            global_max = pos_vals.max()

    elif polarity == DOWN_POLARITY:
        pos_rows, pos_cols = np.where(spikes < 0)
        pos_vals = abs_diff[pos_rows, pos_cols]
        neg_rows = neg_cols = np.array([], dtype=DTYPE_IDX)
        neg_vals = np.array([], dtype=DTYPE)

        if len(pos_vals) > 0:
            global_max = pos_vals.max()

    elif polarity == MERGED_POLARITY:
        neg_rows, neg_cols = np.where(spikes < 0)
        neg_vals = abs_diff[neg_rows, neg_cols]

        pos_rows, pos_cols = np.where(spikes > 0)
        pos_vals = abs_diff[pos_rows, pos_cols]

        if len(neg_vals) > 0 and len(pos_vals) > 0:
            global_max = max(neg_vals.max(), pos_vals.max())
        elif len(neg_vals) > 0:
            global_max = neg_vals.max()
        elif len(pos_vals) > 0:
            global_max = pos_vals.max()

    ####### ROWS, COLS, VALS
    return np.array([neg_rows, neg_cols, neg_vals], dtype=DTYPE), \
        np.array([pos_rows, pos_cols, pos_vals], dtype=DTYPE), \
        global_max


@cython.boundscheck(False)  # turn off bounds-checking for entire function
cdef inline grab_spike_key(DTYPE_IDX_t row, DTYPE_IDX_t col,
                           DTYPE_U8_t flag_shift, DTYPE_U8_t data_shift,
                           DTYPE_U8_t data_mask,
                           DTYPE_U8_t is_pos_spike,
                           DTYPE_U8_t key_coding=KEY_SPINNAKER):
    if key_coding == KEY_SPINNAKER:
        spike_key = spike_to_key(row, col,
                                 flag_shift, data_shift, data_mask,
                                 is_pos_spike)
    elif key_coding == KEY_XYP:
        spike_key = spike_to_xyp(row, col,
                                 is_pos_spike)

    return spike_key

@cython.boundscheck(False)  # turn off bounds-checking for entire function
cdef np.ndarray[DTYPE_t, ndim= 1] spike_to_xyp(DTYPE_IDX_t row, DTYPE_IDX_t col,
                                                DTYPE_U8_t is_pos_spike):
    return np.asarray([col, row, is_pos_spike])


@cython.boundscheck(False)  # turn off bounds-checking for entire function
cdef DTYPE_KEY_t spike_to_key(DTYPE_IDX_t row, DTYPE_IDX_t col,
                          DTYPE_U8_t flag_shift, DTYPE_U8_t data_shift,
                          DTYPE_U8_t data_mask,
                          DTYPE_U8_t is_pos_spike) nogil:
    """
      *nogil allows parallel computation
      Encode row, column and change sign into a 16-bit integer
      :param row:          Row to encode
      :param col:          Column to encode
      :param flag_shift:   How many bits to shift for the pos/neg bit (depends on resolution)
      :param data_shift:   How many bits to shift for the row (depends on resolution)
      :param data_mask:    Bits to take into account for the row/column information
      :param is_pos_spike: A flag to indicate whether the change in brightness was positive
                           or negative
      :returns encoded row/col/pos_or_neg

      The output format is: [col][row][up|down]
    """

    cdef DTYPE_KEY_t d = 0

    # up/down bit
    if is_pos_spike:
        #         d = d | 1 << flag_shift
        d = d | 1

    # col bits
#     d = d | (col)# & data_mask)
    d = d | (row << 1)  # & data_mask)

    # row bits
#     d = d | (row & data_mask) <<  data_shift
    d = d | (col << (data_shift + 1))

    return d


# @cython.boundscheck(False)  # turn off bounds-checking for entire function
# cdef np.ndarray[DTYPE_t, ndim= 2] encode_time_n_bits(np.ndarray[DTYPE_t, ndim=2] in_data,
#                                                       DTYPE_U8_t num_bits):
#     """ Get an approximate value of in_data using only num_bits on bits (matrix version)
#       :param in_data:  Value to encode
#       :param num_bits: Maximum number of bits used in the encoding
#       :returns mult:   Approximate value
#     """
#     cdef np.ndarray[DTYPE_t, ndim = 2] max_pow = DTYPE(np.log2(in_data << 1))
#     cdef np.ndarray[DTYPE_t, ndim= 2] mult = np.power(2, max_pow) >> 1

#     if num_bits == 1:
#         return mult

#     cdef index = 0
#     for index in range(num_bits - 1):
#         # print "encoding cycle = %d"%index
#         # print mult[:10, :10]
#         max_pow = DTYPE(np.log2((in_data - mult) << 1))
#         mult = mult | (np.power(2, max_pow) >> 1)

#     return mult



@cython.boundscheck(False)  # turn off bounds-checking for entire function
def make_spike_lists_rate(np.ndarray[DTYPE_t, ndim=2] pos_spikes,
                          np.ndarray[DTYPE_t, ndim=2] neg_spikes,
                          DTYPE_t global_max,
                          DTYPE_t threshold,
                          DTYPE_U8_t flag_shift,
                          DTYPE_U8_t data_shift,
                          DTYPE_U8_t data_mask,
                          DTYPE_IDX_t max_time_ms,
                          DTYPE_U8_t key_coding=KEY_SPINNAKER):
    """
      Convert spike (row, col, val, sign) lists into a list of Address
      Event Representation (AER) encoded spikes. Rate-encoded values.
      :param pos_spikes:  Positive (up) spikes to encode
      :param neg_spikes:  Negative (down) spikes to encode
      :param global_max:  Maximum change that happened for current frame,
                          used to limit the number of memory slots needed
      :param flag_shift:  How many bits to shift for the pos/neg bit (depends on resolution)
      :param data_shift:  How many bits to shift for the row (depends on resolution)
      :param data_mask:   Bits to take into account for the row/column information
      :param max_time_ms: Upper limit to the number of spikes that can be sent out
      :returns list_of_lists: A list containing lists of keys that should be sent. Each
                              key in the internal lists should be sent "at the same time"
    """
    cdef DTYPE_IDX_t max_spikes = max_time_ms, \
                     len_neg = len(neg_spikes[0]), \
                     len_pos = len(pos_spikes[0])
    cdef DTYPE_IDX_t max_pix = len_neg + len_pos
    cdef DTYPE_IDX_t list_idx

    cdef DTYPE_IDX_t spike_idx, pix_idx, neg_idx
    cdef DTYPE_KEY_t spike_key
    cdef list list_of_lists = list()
    cdef DTYPE_t val

    for list_idx in range(max_spikes):
        list_of_lists.append(list())

    for pix_idx in range(max_pix):
        spike_key = 0

        if pix_idx < len_pos:
            spike_key = grab_spike_key(pos_spikes[ROWS, pix_idx],
                                       pos_spikes[COLS, pix_idx],
                                       flag_shift, data_shift, data_mask,
                                       is_pos_spike=1,
                                       key_coding=key_coding)

            val = pos_spikes[VALS, pix_idx]//threshold
            val = (max_spikes - 1) - val
            spike_idx = DTYPE_IDX(max(0, val))

        else:
            neg_idx = pix_idx - len_pos
            spike_key = grab_spike_key(neg_spikes[ROWS, neg_idx],
                                       neg_spikes[COLS, neg_idx],
                                       flag_shift, data_shift, data_mask,
                                       is_pos_spike=0,
                                       key_coding=key_coding)

            val = neg_spikes[VALS, neg_idx]//threshold
            val = (max_spikes - 1) - val
#~       print("neg rate spikes val, key", val, spike_key)
            spike_idx = DTYPE_IDX(max(0, val))

        for list_idx in range(spike_idx):
            list_of_lists[list_idx].append(spike_key)

    return list_of_lists


@cython.boundscheck(False)  # turn off bounds-checking for entire function
def make_spike_lists_time(np.ndarray[DTYPE_t, ndim=2] pos_spikes,
                          np.ndarray[DTYPE_t, ndim=2] neg_spikes,
                          DTYPE_t global_max,
                          DTYPE_U8_t flag_shift,
                          DTYPE_U8_t data_shift,
                          DTYPE_U8_t data_mask,
                          DTYPE_IDX_t num_bins,
                          DTYPE_t max_time_ms,
                          DTYPE_t min_threshold,
                          DTYPE_t max_threshold,
                          DTYPE_U8_t key_coding=KEY_SPINNAKER):
    """
      Convert spike (row, col, val, sign) lists into a list of Address
      Event Representation (AER) encoded spikes. Time-encoded number of
      rebased thresholds.
      :param pos_spikes:    Positive (up) spikes to encode
      :param neg_spikes:    Negative (down) spikes to encode
      :param global_max:    Maximum change that happened for current frame,
                            used to limit the number of memory slots needed
      :param flag_shift:    How many bits to shift for the pos/neg bit (depends on resolution)
      :param data_shift:    How many bits to shift for the row (depends on resolution)
      :param data_mask:     Bits to take into account for the row/column information
      :param max_time_ms:   Upper limit to the number of spikes that can be sent out
      :param min_threshold: Base threshold for the encoding (just called threshold in other functions)
      :param max_threshold: (Approximate~observed) Maximum change, rounded to next multiple of
                            base threshold
      :returns list_of_lists: A list containing lists of keys that should be sent. Each
                              key in the internal lists should be sent "at the same time"
    """

    cdef unsigned int len_neg = len(neg_spikes[0]), \
        len_pos = len(pos_spikes[0])
    cdef unsigned int max_pix = len_neg + len_pos
    cdef DTYPE_IDX_t num_thresh = 0

    cdef DTYPE_IDX_t time_idx, pix_idx, neg_idx
    cdef DTYPE_t spike_key
    cdef list list_of_lists = list()

    for time_idx in range(num_bins):
        list_of_lists.append(list())

# ~   print list_of_lists

# ~   for pix_idx in prange(max_pix, nogil=True):
    for pix_idx in range(max_pix):
        if pix_idx < len_pos:
            spike_key = grab_spike_key(pos_spikes[ROWS, pix_idx],
                                       pos_spikes[COLS, pix_idx],
                                       flag_shift, data_shift, data_mask,
                                       is_pos_spike=1,
                                       key_coding=key_coding)
            num_thresh = min(pos_spikes[VALS, pix_idx] //
                             min_threshold - 1, num_bins - 1)
        else:
            neg_idx = pix_idx - len_pos
            spike_key = grab_spike_key(neg_spikes[ROWS, neg_idx],
                                       neg_spikes[COLS, neg_idx],
                                       flag_shift, data_shift, data_mask,
                                       is_pos_spike=0,
                                       key_coding=key_coding)

            num_thresh = max(min(neg_spikes[VALS, neg_idx]//min_threshold - 1,
                                 num_bins - 1),
                             0)

#     time_idx = num_thresh
        time_idx = num_bins - num_thresh - 1
# ~     print "num_bins(%s), num_thresh (%s), time_idx (%s)"%(num_bins, num_thresh, time_idx)
        list_of_lists[time_idx].append(spike_key)

    return list_of_lists


@cython.boundscheck(False)  # turn off bounds-checking for entire function
def make_spike_lists_time_bin(np.ndarray[DTYPE_t, ndim=2] pos_spikes,
                              np.ndarray[DTYPE_t, ndim=2] neg_spikes,
                              DTYPE_t global_max,
                              DTYPE_U8_t flag_shift,
                              DTYPE_U8_t data_shift,
                              DTYPE_U8_t data_mask,
                              DTYPE_t max_time_ms,
                              DTYPE_t min_threshold,
                              DTYPE_t max_threshold,
                              DTYPE_t num_bins,
                              np.ndarray[DTYPE_U8_t, ndim=1] log2_table,
                              DTYPE_U8_t key_coding=KEY_SPINNAKER):
    """
      Convert spike (row, col, val, sign) lists into a list of Address
      Event Representation (AER) encoded spikes. Time/binary-encoded difference
      value.
      :param pos_spikes:    Positive (up) spikes to encode
      :param neg_spikes:    Negative (down) spikes to encode
      :param global_max:    Maximum change that happened for current frame,
                            used to limit the number of memory slots needed
      :param flag_shift:    How many bits to shift for the pos/neg bit (depends on resolution)
      :param data_shift:    How many bits to shift for the row (depends on resolution)
      :param data_mask:     Bits to take into account for the row/column information
      :param max_time_ms:   Upper limit to the number of spikes that can be sent out
      :param min_threshold: Base threshold for the encoding (just called threshold in other functions)
      :param max_threshold: (Approximate~observed) Maximum change, rounded to next multiple of
                            base threshold
      :param num_bits:      Number of active bits in the representation.
      :returns list_of_lists: A list containing lists of keys that should be sent. Each
                              key in the internal lists should be sent "at the same time"
    """
    cdef unsigned int len_neg = len(neg_spikes[0]), \
        len_pos = len(pos_spikes[0])
    cdef unsigned int max_pix = len_neg + len_pos

    cdef np.ndarray[DTYPE_IDX_t, ndim= 1] indices
    cdef DTYPE_IDX_t time_idx, pix_idx, neg_idx
    cdef DTYPE_t spike_key
    cdef list list_of_lists = list()
    cdef DTYPE_U8_t byte_code

    # 8-bit images
    for spike_key in range(num_bins):
        list_of_lists.append(list())

# ~   for pix_idx in prange(max_pix, nogil=True):
    for pix_idx in range(max_pix):
        if pix_idx < len_pos:
            spike_key = grab_spike_key(pos_spikes[ROWS, pix_idx],
                                       pos_spikes[COLS, pix_idx],
                                       flag_shift, data_shift, data_mask,
                                       is_pos_spike=1,
                                       key_coding=key_coding)

            byte_code = log2_table[pos_spikes[VALS, pix_idx]]

            indices, = np.where(np.unpackbits(np.uint8(byte_code)))

            for i in indices:
                list_of_lists[i].append(spike_key)

        else:
            neg_idx = pix_idx - len_pos
            spike_key = grab_spike_key(neg_spikes[ROWS, neg_idx],
                                       neg_spikes[COLS, neg_idx],
                                       flag_shift, data_shift, data_mask,
                                       is_pos_spike=0,
                                       key_coding=key_coding)

            byte_code = log2_table[neg_spikes[VALS, neg_idx]]

            indices, = np.where(np.unpackbits(np.uint8(byte_code)))

            for i in indices:
                list_of_lists[i].append(spike_key)

    return list_of_lists

import numpy as np
cimport numpy as np
cimport cython

from pydvs.cdefines cimport *
from pydvs.pdefines import *

@cython.boundscheck(False)  # turn off bounds-checking for entire function
cdef inline DTYPE_IDX_t argmax(DTYPE_t[:] ary, DTYPE_IDX_t width) nogil:
    """
      The *nogil* modifier allows to be computed in parallel outside the function
      :param ary:   Array where the maximum is to be found
      :param width: Width of the array
      :returns idx: Index of the maximum value in an array
    """
    cdef DTYPE_IDX_t idx = 0
    cdef DTYPE_t val = ary[idx]
    for i in range(1, width):
        if ary[i] > val:
            idx = i
            val = ary[i]

    return idx


@cython.boundscheck(False)  # turn off bounds-checking for entire function
cdef inline transform_coords(
    DTYPE_IDX_t u, DTYPE_IDX_t v, DTYPE_IDX_t inh_w, DTYPE_IDX_t ratio):
    """ Maps line to 2D area coordinates
        ------------
        | A | B |...         -----------------
        ------------   <==   | A | B | C | D |
        | C | D |...         -----------------
        ------------         | : | : | : | : |
        | : | : |.:.

      :param u:       Vertical coordinate (row index)
      :param v:       Horizontal coordinate (column index)
      :param inh_w:   Area width
      :param ratio:   Total image width to area width ratio
      :returns i, j:  Area coordinates
    """
    return inh_w*(u//ratio) + v//inh_w, inh_w*(u % ratio) + v % inh_w


@cython.boundscheck(False)  # turn off bounds-checking for entire function
cdef inline generate_inh_coords(
            DTYPE_IDX_t width, DTYPE_IDX_t height, DTYPE_IDX_t inh_width):
    """
      Maps all the coordinates of lines to areas in a 2D array, used to precompute
      the transformation.
      See "transform_coords(u, v, inh_w, ratio)"
      :params width:    Image width
      :params height:   Image height
      :param inh_width: Area width
      :returns coords:  An array of transformed coordinates [[row0, col0]...[rowN, colN]]
    """
    cdef DTYPE_IDX_t u, v, new_w, new_h, ratio
    new_w = inh_width*inh_width
    new_h = (height*width)//new_w
    ratio = width//inh_width

    return np.asarray([transform_coords(u, v, inh_width, ratio)
                        for u in range(new_h)
                            for v in range(new_w)],
                      dtype=DTYPE)

@cython.boundscheck(False)  # turn off bounds-checking for entire function
cdef inline DTYPE_U8_t encode_time_n_bits_single(DTYPE_t in_data, DTYPE_U8_t num_bits):
    """ Get an approximate value of in_data using only num_bits on bits (scalar version)
      :param in_data:  Value to encode
      :param num_bits: Maximum number of bits used in the encoding
      :returns mult:   Approximate value
    """
    if in_data == 0:
        return 0

    cdef DTYPE_t max_pow = DTYPE(np.log2(in_data))
    cdef DTYPE_U8_t mult = DTYPE_U8(np.power(2, max_pow))

    if num_bits == 1:
        return mult

    cdef int index = 0
    for index in range(num_bits - 1):
        if (in_data - mult) > 0:
            max_pow = DTYPE_(np.log2(in_data - mult))
            mult = mult | DTYPE_U8(np.power(2, max_pow))
        else:
            break

    return mult

@cython.boundscheck(False) 
cdef inline generate_log2_table(max_active_bits, bit_resolution):
    """Create a look-up table for the possible values in the range (0, 2^bit_resolution)
      one table per active bits in the range (0, max_active_bits]
    """
    cdef np.ndarray[DTYPE_U8_t, ndim= 2] log2_table = \
                np.zeros((max_active_bits, 2**bit_resolution), dtype=DTYPE_U8)
    cdef int active_bits, value

    for active_bits in range(max_active_bits):
        for value in np.arange(2**bit_resolution, dtype=DTYPE):

            log2_table[active_bits][value] = encode_time_n_bits_single(
                value, active_bits)

    return log2_table




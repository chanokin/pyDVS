import numpy as np
cimport numpy as np
cimport cython

from pydvs.cdefines cimport *
from pydvs.pdefines import *

@cython.boundscheck(False)  # turn off bounds-checking for entire function
cdef root_mean_square(np.ndarray[DTYPE_t, ndim=2] original,
                             np.ndarray[DTYPE_t, ndim=2] estimate):
    """
      Calculate the root of the mean of the squared difference
    """

    return np.sqrt(((original - estimate)**2).mean())

@cython.boundscheck(False)  # turn off bounds-checking for entire function
cdef average_bits(np.ndarray[DTYPE_U8_t, ndim=2] num_bits):
    """
      Get the average number of bits in an array
    """
    return num_bits[np.where(num_bits > 0)].mean()


@cython.boundscheck(False)  # turn off bounds-checking for entire function
cdef count_bits(np.ndarray[DTYPE_t, ndim=2] in_data):
    """Count the number of active bits in in_data
    """

    cdef np.ndarray[DTYPE_U8_t, ndim= 1] bits = np.unpackbits(in_data.astype(DTYPE_U8))
    return bits.reshape((bits.size//16, 2, 8)).sum(axis=2).astype(DTYPE)

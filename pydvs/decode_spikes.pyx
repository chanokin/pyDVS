import numpy
cimport numpy

import cv2

cimport cython

from cpython cimport array
import array

from cython.parallel import prange

try:                  
  from cv2.cv import CV_INTER_AREA 
except:
  from cv2 import INTER_AREA as CV_INTER_AREA 
  
DTYPE = numpy.int16
ctypedef numpy.int16_t DTYPE_t
DTYPE_str = 'h'

DTYPE_IDX = numpy.int64
ctypedef numpy.int64_t DTYPE_IDX_t

DTYPE_U8 = numpy.uint8
ctypedef numpy.uint8_t DTYPE_U8_t

DTYPE_U16 = numpy.uint16
ctypedef numpy.uint16_t DTYPE_U16_t

DTYPE_U32 = numpy.uint32
ctypedef numpy.uint32_t DTYPE_U32_t

DTYPE_FLOAT = numpy.float
ctypedef numpy.float_t DTYPE_FLOAT_t

DEF UP_POLARITY    = 0
DEF DOWN_POLARITY  = 1
DEF MERGED_POLARITY = 2
DEF VALS = 2
DEF COLS = 1
DEF ROWS = 0


@cython.boundscheck(False) # turn off bounds-checking for entire function
cdef key_to_spike(DTYPE_U16_t key,
                  DTYPE_U8_t flag_shift, DTYPE_U8_t data_shift, 
                  DTYPE_U8_t data_mask):
    """
      *nogil allows parallel computation
      Decode row, column and change sign from a 16-bit integer
      :param key:          Key to decode
      :param flag_shift:   How many bits to shift for the pos/neg bit (depends on resolution)
      :param data_shift:   How many bits to shift for the row (depends on resolution)
      :param data_mask:    Bits to take into account for the row/column information
      :returns decoded:    [row, column, positive or negative one]
    """
    cdef DTYPE_t row  = (key >> data_shift) & data_mask
    cdef DTYPE_t col  = key & data_mask
    cdef DTYPE_t sign = 0

    if ((key >> flag_shift) & 1) == 0:
      sign = 1
    else:
      sign = -1
    
    return row, col, sign
    
    

@cython.boundscheck(False) # turn off bounds-checking for entire function
def update_ref_spike_list_rate(numpy.ndarray[DTYPE_t, ndim=2] ref,
                               numpy.ndarray[DTYPE_t, ndim=1] spikes,
                               DTYPE_t threshold,
                               DTYPE_FLOAT_t history_weight,
                               DTYPE_U8_t flag_shift, DTYPE_U8_t data_shift, 
                               DTYPE_U8_t data_mask):
  cdef int idx = 0
  cdef num_spikes = len(spikes)
  cdef DTYPE_U16_t key
  cdef DTYPE_t row
  cdef DTYPE_t col
  cdef DTYPE_t sign
  
  ref = DTYPE(ref*history_weight)
  
#~   for idx in prange(num_spikes, nogil=True):
  for idx in range(num_spikes):
    row, col, sign = key_to_spike(spikes[idx], flag_shift, data_shift, data_mask)
    ref[row, col] = ref[row, col] + threshold*sign

  return ref



@cython.boundscheck(False) # turn off bounds-checking for entire function
def update_ref_spike_list_time(numpy.ndarray[DTYPE_t, ndim=2] ref,
                               numpy.ndarray[DTYPE_t, ndim=2] time_last,
                               numpy.ndarray[DTYPE_t, ndim=2] val_last,
                               numpy.ndarray[DTYPE_t, ndim=1] spikes,
                               DTYPE_t num_bins, 
                               DTYPE_FLOAT_t bin_width,
                               DTYPE_t time_period,
                               DTYPE_U32_t time_now,
                               DTYPE_t threshold,
                               DTYPE_FLOAT_t history_weight,
                               DTYPE_U8_t flag_shift, DTYPE_U8_t data_shift, 
                               DTYPE_U8_t data_mask):
  cdef int idx = 0
  cdef int num_spikes = len(spikes)
  cdef int num_thresh = 0
  cdef DTYPE_FLOAT_t time_diff = 0
  cdef DTYPE_t val_now = 0
  cdef DTYPE_U16_t key
  cdef DTYPE_t row
  cdef DTYPE_t col
  cdef DTYPE_t sign
  
  ref = DTYPE(ref*history_weight)
  
  
  for idx in range(num_spikes):
    
    row, col, sign = key_to_spike(spikes[idx], flag_shift, data_shift, data_mask)
    
    if val_last[row, col] == -1: # first time val_now is always 1
#~       print "(%s, %s) <= ref + update = %s + %s"%\
#~              (row, col, ref[row, col], threshold+1)
      ref[row, col] = ref[row, col] + (threshold+1)#*1 
      val_last[row, col] = threshold +1
      time_last[row, col] = time_now

    else:  
      time_diff = time_now - (time_last[row, col] - (num_bins - val_last[row, col]*bin_width))
      val_now = numpy.floor(num_bins - numpy.mod(time_diff, time_period)/bin_width)
#~       print "(%s, %s) <= ref + update = %s + %s"%\
#~              (row, col, ref[row, col], threshold*sign*val_now)
             
      ref[row, col] = ref[row, col] + threshold*sign*val_now
      
      val_last[row, col] = threshold*sign*val_now
      time_last[row, col] = time_now


  return ref, time_last, val_last



@cython.boundscheck(False) # turn off bounds-checking for entire function
def update_ref_spike_list_time_bin(numpy.ndarray[DTYPE_t, ndim=2] ref,
                                   numpy.ndarray[DTYPE_t, ndim=2] time_last,
                                   numpy.ndarray[DTYPE_t, ndim=2] val_last,
                                   numpy.ndarray[DTYPE_t, ndim=1] spikes,
                                   DTYPE_t num_bins, 
                                   DTYPE_FLOAT_t bin_width,
                                   DTYPE_t time_period,
                                   DTYPE_U32_t time_now,
                                   DTYPE_t threshold,
                                   DTYPE_FLOAT_t history_weight,
                                   DTYPE_U8_t flag_shift, DTYPE_U8_t data_shift, 
                                   DTYPE_U8_t data_mask):
  cdef int idx = 0
  cdef int num_spikes = len(spikes)
  cdef int num_thresh = 0
  cdef DTYPE_FLOAT_t time_diff = 0
  cdef DTYPE_t val_now = 0
  cdef DTYPE_U16_t key
  cdef DTYPE_t row
  cdef DTYPE_t col
  cdef DTYPE_t sign
  
  ref = DTYPE(ref*history_weight)
  
  for idx in range(num_spikes):
    
    row, col, sign = key_to_spike(spikes[idx], flag_shift, data_shift, data_mask)
    
    if val_last[row, col] == -1: # first time val_now is always 1
      ref[row, col] = ref[row, col] + (threshold + 1)*sign#*2**0 
      
      val_last[row, col] = (threshold + 1)*sign#*2**0 
      time_last[row, col] = time_now

      continue
    else:  
      time_diff = time_now - \
                  (time_last[row, col] - 2*(num_bins - numpy.log2(val_last[row, col])*bin_width + 1))

      val_now = numpy.power(2, 
                            numpy.floor(num_bins - numpy.mod(time_diff, time_period)/bin_width) )
    
      ref[row, col] = ref[row, col] + threshold*sign*val_now

      val_last[row, col] = threshold*sign*val_now
      time_last[row, col] = time_now

  return ref, time_last, val_last



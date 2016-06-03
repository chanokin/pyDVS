import numpy
cimport numpy

import cv2
import time

cimport cython
# from cpython.type import Py_ssize
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

DTYPE_FLOAT = numpy.float
ctypedef numpy.float_t DTYPE_FLOAT_t

DEF UP_POLARITY    = 0
DEF DOWN_POLARITY  = 1
DEF MERGED_POLARITY = 2
DEF RECTIFIED_POLARITY = 3
DEF VALS = 2
DEF COLS = 1
DEF ROWS = 0

DEF KEY_SPINNAKER = 0
DEF KEY_XYP = 1


@cython.boundscheck(False) # turn off bounds-checking for entire function
def thresholded_difference(numpy.ndarray[DTYPE_t, ndim=2] curr_frame,
                           numpy.ndarray[DTYPE_t, ndim=2] ref_frame,
                           DTYPE_t threshold):
  """
    :param curr_frame: Latest image captured by the camera
    :param ref_frame:  Saves value when the pixel was marked as "spiking"
    :param threshold:  How big the difference between current and reference
                       frames needs to be to mark a pixel as spiking

    :return diff:     Signed difference between current and reference frames
    :return abs_diff: Absolute value of the difference
    :return spikes:   Signed pixels marked as spiking (-1 means change from higher
                      to lower brightness value, 1 means change from lower to
                      higher brightness value.)
  """
  cdef numpy.ndarray[DTYPE_t, ndim=2] diff, abs_diff, spikes
  cdef numpy.ndarray[DTYPE_IDX_t, ndim=1] neg_r, neg_c

  diff = curr_frame - ref_frame
  abs_diff = numpy.abs(diff)

  spikes = (abs_diff > threshold).astype(DTYPE)

  abs_diff = (abs_diff*spikes)

  neg_r, neg_c = numpy.where(diff < -threshold)
  spikes[neg_r, neg_c] = -1


  return diff, abs_diff, spikes



@cython.boundscheck(False) # turn off bounds-checking for entire function
def thresholded_difference_adpt(numpy.ndarray[DTYPE_t, ndim=2] curr_frame,
                                numpy.ndarray[DTYPE_t, ndim=2] ref_frame,
                                numpy.ndarray[DTYPE_t, ndim=2] threshold,
                                DTYPE_t min_threshold,
                                DTYPE_t max_threshold):
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
  cdef numpy.ndarray[DTYPE_t, ndim=2] diff, abs_diff, spikes
  cdef numpy.ndarray[DTYPE_IDX_t, ndim=1] neg_r, neg_c

  diff = curr_frame - ref_frame
  abs_diff = numpy.abs(diff)


  spikes = (abs_diff > threshold).astype(DTYPE)
  abs_diff = (abs_diff*spikes)
  neg_r, neg_c = numpy.where(diff < -threshold)
  spikes[neg_r, neg_c] = -1

  return diff, abs_diff, spikes



@cython.boundscheck(False) # turn off bounds-checking for entire function
cdef Py_ssize_t argmax(DTYPE_t[:] ary, DTYPE_t width) nogil:
  """
    The *nogil* modifier allows to be computed in parallel outside the function
    :param ary:   Array where the maximum is to be found
    :param width: Width of the array
    :returns idx: Index of the maximum value in an array
  """
  cdef Py_ssize_t idx = 0;
  cdef DTYPE_t val = ary[idx];
  for i in range(1, width):
    if ary[i] > val:
      idx = i
      val = ary[i]

  return idx


@cython.boundscheck(False) # turn off bounds-checking for entire function
cdef inline transform_coords(u, v, inh_w, ratio):
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
  return inh_w*(u/ratio) + v/inh_w, inh_w*(u%ratio) + v%inh_w


@cython.boundscheck(False) # turn off bounds-checking for entire function
def generate_inh_coords(DTYPE_t width, DTYPE_t height, DTYPE_t inh_width):
  """
    Maps all the coordinates of lines to areas in a 2D array, used to precompute
    the transformation.
    See "transform_coords(u, v, inh_w, ratio)"
    :params width:    Image width
    :params height:   Image height
    :param inh_width: Area width
    :returns coords:  An array of transformed coordinates [[row0, col0]...[rowN, colN]]
  """
  cdef Py_ssize_t u, v, new_w, new_h, ratio
  new_w = inh_width*inh_width
  new_h = (height*width)/new_w
  ratio = width/inh_width

  return numpy.array([transform_coords(u, v, inh_width, ratio) \
                                        for u in range(new_h) \
                                        for v in range(new_w)],
                    dtype=DTYPE)


@cython.boundscheck(False) # turn off bounds-checking for entire function
def local_inhibition(numpy.ndarray[DTYPE_t, ndim=2] spikes,
                     numpy.ndarray[DTYPE_t, ndim=2] abs_diff,
                     numpy.ndarray[DTYPE_t, ndim=2] inh_coords,
                     DTYPE_t width,
                     DTYPE_t height,
                     DTYPE_t inh_width):
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
  cdef Py_ssize_t u, v, new_w, new_h, ratio

#~   cdef numpy.ndarray[DTYPE_t, ndim=2] reshaped
  cdef DTYPE_t[:,:] reshaped
  cdef numpy.ndarray[DTYPE_t, ndim=1] max_vals
  cdef numpy.ndarray[Py_ssize_t, ndim=2] max_indices
  cdef numpy.ndarray[Py_ssize_t, ndim=1] max_local_indices


  new_w = inh_width*inh_width
  new_h = (height*width)/new_w
  ratio = width/inh_width

  #(w*h/inh_w^2, inh_w^2) so we can efficiently find local max
  reshaped = abs_diff[inh_coords[:, ROWS], inh_coords[:, COLS]].reshape((new_h, new_w))
  max_local_indices = numpy.zeros(new_h, dtype=DTYPE_IDX)
  #allow parallel compute of max indices
  for v in prange(new_h, nogil=True):
    max_local_indices[v] = argmax(reshaped[v], new_w)

  max_indices = numpy.array([transform_coords(u, max_local_indices[u], inh_width, ratio) \
                             for u in range(new_h)], dtype=DTYPE_IDX)

  max_vals = spikes[max_indices[:, ROWS], max_indices[:, COLS]]
  spikes[:] = 0
  spikes[max_indices[:, ROWS], max_indices[:, COLS]] = max_vals


  return spikes


@cython.boundscheck(False) # turn off bounds-checking for entire function
def update_reference_rate(numpy.ndarray[DTYPE_t, ndim=2] abs_diff,
                          numpy.ndarray[DTYPE_t, ndim=2] spikes,
                          numpy.ndarray[DTYPE_t, ndim=2] ref_frame,
                          DTYPE_t threshold,
                          DTYPE_t max_time_ms,
                          DTYPE_FLOAT_t history_weight):
  """
    Rate based spike transmission.
    :param abs_diff:        Absolute value of the difference of current frame and
                            reference frame (computed in *thresholded_difference*)
    :param spikes:          Pixels marked as spiking
    :param ref_frame:       Previous reference frame
    :param threshold:       How much brightness has to change to mark a pixel as spiking
    :param max_time_ms:     Number of milliseconds between each frame (saturation limit)
    :param history_weight:  How much does the previous reference frame weighs in the
                            update equation
    :returns ref_frame:     Updated reference frame
  """

  cdef numpy.ndarray[DTYPE_t, ndim=2] num_spikes = abs_diff/threshold

  # at most max_time_ms spikes
  num_spikes = numpy.clip(num_spikes, 0, max_time_ms, out=num_spikes)
  # can only update N*threshold quantities
  ref_frame = numpy.clip( DTYPE(history_weight*ref_frame) + num_spikes*spikes*threshold, 0, 255)

  return ref_frame



@cython.boundscheck(False) # turn off bounds-checking for entire function
def update_reference_rate_adpt(numpy.ndarray[DTYPE_t, ndim=2] abs_diff,
                               numpy.ndarray[DTYPE_t, ndim=2] spikes,
                               numpy.ndarray[DTYPE_t, ndim=2] ref_frame,
                               numpy.ndarray[DTYPE_t, ndim=2] threshold,
                               DTYPE_t min_threshold,
                               DTYPE_t max_threshold,
                               DTYPE_t down_threshold_change,
                               DTYPE_t up_threshold_change,
                               DTYPE_t max_time_ms,
                               DTYPE_FLOAT_t history_weight):
  """
    Rate based spike transmission with adaptive threshold.
    :param abs_diff:        Absolute value of the difference of current frame and
                            reference frame (computed in *thresholded_difference*)
    :param spikes:          Pixels marked as spiking
    :param ref_frame:       Previous reference frame
    :param threshold:       How much brightness has to change to mark a pixel as spiking in
                            a per-pixel basis. Increased if the pixel was marked as spiking,
                            decrease otherwise.
    :param min_threshold:   Lower cap for the threshold variation
    :param max_threshold:   Upper cap for the threshold variation
    :param down_threshold_change: How much should the threshold change per frame, when a pixel didn't spike (-)
    :param up_threshold_change:   How much should the threshold change per frame, when a pixel spiked (+)
    :param max_time_ms:     Number of milliseconds between each frame (saturation limit)
    :param history_weight:  How much does the previous reference frame weighs in the
                            update equation
    :returns ref_frame:     Updated reference frame
    :returns threshold:     Updated per-pixel threshold matrix
  """
  ### can only update N*threshold quantities
  cdef numpy.ndarray[DTYPE_t, ndim=2] update = numpy.zeros_like(spikes, dtype=DTYPE)
  cdef numpy.ndarray[DTYPE_t, ndim=2] num_spikes = abs_diff/threshold

  ref_frame = DTYPE(history_weight*ref_frame) + num_spikes*spikes*min_threshold

  update = numpy.logical_and(spikes != 0, threshold < max_threshold).astype(dtype=DTYPE)*\
                                                                       up_threshold_change

  threshold += update

  update = numpy.logical_and(spikes == 0, threshold > min_threshold).astype(dtype=DTYPE)*\
                                                                     down_threshold_change
  threshold -= update

  return numpy.clip(ref_frame, 0, 255), threshold


@cython.boundscheck(False) # turn off bounds-checking for entire function
def update_reference_time_thresh(numpy.ndarray[DTYPE_t, ndim=2] abs_diff,
                                 numpy.ndarray[DTYPE_t, ndim=2] spikes,
                                 numpy.ndarray[DTYPE_t, ndim=2] ref_frame,
                                 DTYPE_t threshold,
                                 DTYPE_t max_time_ms,
                                 DTYPE_FLOAT_t history_weight):
  """
    Time based spike transmission.
    :param abs_diff:        Absolute value of the difference of current frame and
                            reference frame (computed in *thresholded_difference*)
    :param spikes:          Pixels marked as spiking
    :param ref_frame:       Previous reference frame
    :param threshold:       How much brightness has to change to mark a pixel as spiking
    :param max_time_ms:     Number of milliseconds between each frame (saturation limit)
    :param history_weight:  How much does the previous reference frame weighs in the
                            update equation
    :returns ref_frame:     Updated reference frame

    Linear mapping of the number of thresholds rebased by the frame difference

    TH <- Threshold value
    T  <- Inter-frame time (1/frames per second)
    n  <- min(MAX_THRESH_CHANGES, Inter-frame time)

   t=0 _______________________|____________ t=T
       |      |      |      |      |      |
         nTH  (n-1)TH  ...    2TH     TH

  """
  cdef numpy.ndarray[DTYPE_t, ndim=2] mult = numpy.clip(abs_diff/threshold, 0, max_time_ms)
#~   print abs_diff[:10, :10]
  ref_frame = numpy.clip( DTYPE(history_weight*ref_frame) + spikes*mult*threshold, 0, 255)

  return ref_frame



@cython.boundscheck(False) # turn off bounds-checking for entire function
def update_reference_time_binary_raw(numpy.ndarray[DTYPE_t, ndim=2] abs_diff,
                                     numpy.ndarray[DTYPE_t, ndim=2] spikes,
                                     numpy.ndarray[DTYPE_t, ndim=2] ref_frame,
                                     DTYPE_t threshold,
                                     DTYPE_t max_time_ms,
                                     DTYPE_t num_spikes,
                                     DTYPE_FLOAT_t history_weight,
                                     numpy.ndarray[DTYPE_U8_t, ndim=1] log2_table):
  """
    Time based spike transmission.
    :param abs_diff:        Absolute value of the difference of current frame and
                            reference frame (computed in *thresholded_difference*)
    :param spikes:          Pixels marked as spiking
    :param ref_frame:       Previous reference frame
    :param threshold:       How much brightness has to change to mark a pixel as spiking
    :param max_time_ms:     Number of milliseconds between each frame (saturation limit)
    :param num_spikes:      Number of active bits to use in the encoding
    :param history_weight:  How much does the previous reference frame weighs in the
                            update equation
    :param log2_table:      Precomputed versions of the raw difference using only num_spikes
                            active bits
    :returns ref_frame:     Updated reference frame

    Binary encoding of the raw brightness difference value, using only num_spikes active bits
    (at most 8-bit precision needed)

    TH <- Threshold value
    T  <- Inter frame time (1/frames per second)

   t=0 __________|___________|__________ t=T
       |   |   |   |   |   |   |   |   |
   bit   8   7   6   5   4   3   2   1
   val           32          4
   original = 38, encoded with only 2 active bits = 36
  """
  cdef numpy.ndarray[DTYPE_t, ndim=2] mult = DTYPE(log2_table[abs_diff])
#~   print abs_diff[:10, :10]
  ref_frame = numpy.clip( DTYPE(history_weight*ref_frame) + spikes*mult, 0, 255)

  return ref_frame


@cython.boundscheck(False) # turn off bounds-checking for entire function
def update_reference_time_binary_thresh(numpy.ndarray[DTYPE_t, ndim=2] abs_diff,
                                        numpy.ndarray[DTYPE_t, ndim=2] spikes,
                                        numpy.ndarray[DTYPE_t, ndim=2] ref_frame,
                                        DTYPE_t threshold,
                                        DTYPE_t max_time_ms,
                                        DTYPE_t num_spikes,
                                        DTYPE_FLOAT_t history_weight,
                                        numpy.ndarray[DTYPE_U8_t, ndim=1] log2_table):

  """
    Time based spike transmission.
    :param abs_diff:        Absolute value of the difference of current frame and
                            reference frame (computed in *thresholded_difference*)
    :param spikes:          Pixels marked as spiking
    :param ref_frame:       Previous reference frame
    :param threshold:       How much brightness has to change to mark a pixel as spiking
    :param max_time_ms:     Number of milliseconds between each frame (saturation limit)
    :param num_spikes:      Number of active bits to use in the encoding
    :param history_weight:  How much does the previous reference frame weighs in the
                            update equation
    :param log2_table:      Precomputed versions of the raw difference using only num_spikes
                            active bits
    :returns ref_frame:     Updated reference frame

    Binary encoding of the number of thresholds rebased by the brightness difference,
    using only num_spikes active bits

    TH <- Threshold value
    T  <- Inter frame time (1/frames per second)

   t=0 _______________|_____|________ t=T
       |     |     |     |     |     |
   bit    5     4     3     2     1
   val                8     2
   original = 125, threshold = 12 => ~10 thresholds
   encoded with only 2 active bits = 10*12 = 120

  """

  cdef numpy.ndarray[DTYPE_t, ndim=2] mult = DTYPE(log2_table[abs_diff/threshold])

  ref_frame = numpy.clip( DTYPE(history_weight*ref_frame) + spikes*mult*threshold, 0, 255)

  return ref_frame

@cython.boundscheck(False) # turn off bounds-checking for entire function
def update_reference_time_binary_raw_noise(numpy.ndarray[DTYPE_t, ndim=2] abs_diff,
                                           numpy.ndarray[DTYPE_t, ndim=2] spikes,
                                           numpy.ndarray[DTYPE_t, ndim=2] ref_frame,
                                           DTYPE_t width,
                                           DTYPE_t height,
                                           DTYPE_t threshold,
                                           DTYPE_t max_time_ms,
                                           DTYPE_t num_spikes,
                                           DTYPE_FLOAT_t history_weight,
                                           numpy.ndarray[DTYPE_U8_t, ndim=1] log2_table,
                                           DTYPE_t noise_probability):

  cdef numpy.ndarray[DTYPE_t, ndim=2] mult = DTYPE(log2_table[abs_diff])
  cdef int num_indices = int((float(noise_probability)/100.)*(width*height))
  cdef numpy.ndarray[DTYPE_t, ndim=1] sample = DTYPE(numpy.random.randint(0, height*width, size=num_indices))
  cdef numpy.ndarray[DTYPE_t, ndim=1] rows = sample/width
  cdef numpy.ndarray[DTYPE_t, ndim=1] cols = sample%width
  mult[rows,cols] = mult[rows,cols] ^ 0xFF # add noise by xor with 1111 1111
#~   print abs_diff[:10, :10]
  ref_frame = DTYPE(history_weight*ref_frame) + spikes*mult

  return ref_frame


@cython.boundscheck(False) # turn off bounds-checking for entire function
def update_reference_time_thresh_noise(numpy.ndarray[DTYPE_t, ndim=2] abs_diff,
                                       numpy.ndarray[DTYPE_t, ndim=2] spikes,
                                       numpy.ndarray[DTYPE_t, ndim=2] ref_frame,
                                       DTYPE_t width,
                                       DTYPE_t height,
                                       DTYPE_t threshold,
                                       DTYPE_t max_time_ms,
                                       DTYPE_FLOAT_t history_weight,
                                       DTYPE_t noise_probability):

  cdef numpy.ndarray[DTYPE_t, ndim=2] mult = numpy.clip(abs_diff/threshold, 0, max_time_ms)
  cdef int num_indices = int((float(noise_probability)/100.)*(width*height))
  cdef numpy.ndarray[DTYPE_t, ndim=1] sample = DTYPE(numpy.random.randint(0, height*width, size=num_indices))
  cdef numpy.ndarray[DTYPE_t, ndim=1] rows = sample/width
  cdef numpy.ndarray[DTYPE_t, ndim=1] cols = sample%width
#   mult[rows,cols] = mult[rows,cols] ^ 0xFF # add noise by xor with 1111 1111
  mult[rows,cols] = 0
#~   print abs_diff[:10, :10]
  ref_frame = DTYPE(history_weight*ref_frame) + spikes*mult

  return ref_frame


@cython.boundscheck(False) # turn off bounds-checking for entire function
cdef DTYPE_U8_t encode_time_n_bits_single(DTYPE_t in_data, DTYPE_t num_bits):
  """ Get an approximate value of in_data using only num_bits on bits (scalar version)
    :param in_data:  Value to encode
    :param num_bits: Maximum number of bits used in the encoding
    :returns mult:   Approximate value
  """
  if in_data == 0:
    return 0

  cdef DTYPE_t max_pow = DTYPE(numpy.log2(in_data))
  cdef DTYPE_U8_t mult = DTYPE_U8(numpy.power(2, max_pow))

  if num_bits == 1:
    return mult

  cdef int index = 0;
  for index in range(num_bits - 1):
    if (in_data - mult) > 0:
      max_pow = DTYPE( numpy.log2(in_data - mult) )
      mult = mult | DTYPE_U8( numpy.power(2, max_pow) )
    else:
      break

  return mult



@cython.boundscheck(False) # turn off bounds-checking for entire function
cdef numpy.ndarray[DTYPE_t, ndim=2] encode_time_n_bits(numpy.ndarray[DTYPE_t, ndim=2] in_data,
                                                       DTYPE_t num_bits):
  """ Get an approximate value of in_data using only num_bits on bits (matrix version)
    :param in_data:  Value to encode
    :param num_bits: Maximum number of bits used in the encoding
    :returns mult:   Approximate value
  """
  cdef numpy.ndarray[DTYPE_t, ndim=2] max_pow = DTYPE(numpy.log2(in_data <<1))
  cdef numpy.ndarray[DTYPE_t, ndim=2] mult = numpy.power(2, max_pow) >> 1

  if num_bits == 1:
    return mult

  cdef index = 0;
  for index in range(num_bits - 1):
    #print "encoding cycle = %d"%index
    #print mult[:10, :10]
    max_pow = DTYPE( numpy.log2((in_data - mult) << 1) )
    mult = mult | ( numpy.power(2, max_pow) >> 1 )

  return mult



@cython.boundscheck(False) # turn off bounds-checking for entire function
def count_bits(numpy.ndarray[DTYPE_t, ndim=2] in_data):
  """Count the number of active bits in in_data
  """

  cdef numpy.ndarray[DTYPE_U8_t, ndim=1] bits = numpy.unpackbits(in_data.astype(DTYPE_U8))
  return bits.reshape((bits.size/16, 2, 8)).sum(axis=2).astype(DTYPE)




@cython.boundscheck(False) # turn off bounds-checking for entire function
def average_bits(numpy.ndarray[DTYPE_t, ndim=2] num_bits):
  """
    Get the average number of bits in an array
  """
  return num_bits[numpy.where(num_bits > 0)].mean()



@cython.boundscheck(False) # turn off bounds-checking for entire function
def root_mean_square(numpy.ndarray[DTYPE_t, ndim=2] original,
                     numpy.ndarray[DTYPE_t, ndim=2] estimate):
  """
    Calculate the root of the mean of the squared difference
  """
  cdef numpy.ndarray[DTYPE_FLOAT_t, ndim=2] orig = original*1.0
  cdef numpy.ndarray[DTYPE_FLOAT_t, ndim=2] esti = estimate*1.0

  return numpy.sqrt( ((orig - esti)**2).mean() )






@cython.boundscheck(False) # turn off bounds-checking for entire function
def render_frame(numpy.ndarray[DTYPE_t, ndim=2] spikes,
                 numpy.ndarray[DTYPE_t, ndim=2] curr_frame,
                 DTYPE_t width,
                 DTYPE_t height,
                 DTYPE_U8_t polarity):
  """
    Overlaps the generated spikes onto the latest image from the video
    source. Red means a negative change in brightness, Green a positive one.

    :param spikes:     Pixels marked as spiking
    :param curr_frame: Latest image from the video source
    :param width:      Image width
    :param height:     Image height
    :param polarity:   Wether to report positive, negative or both changes
                       in brightness
    :returns spikes_frame: Combined spikes/image information in a color image
  """
  cdef numpy.ndarray[DTYPE_U8_t, ndim=3] spikes_frame = numpy.zeros([height, width, 3], dtype=DTYPE_U8)
  cdef numpy.ndarray[Py_ssize_t, ndim=1] rows, cols
  spikes_frame[:, :, 0] = curr_frame
  spikes_frame[:, :, 1] = curr_frame
  spikes_frame[:, :, 2] = curr_frame

  if polarity == RECTIFIED_POLARITY:
    rows, cols = numpy.where(spikes != 0)
    spikes_frame[rows, cols, :] = [0, 255, 0]

  if polarity == UP_POLARITY or polarity == MERGED_POLARITY:
    rows, cols = numpy.where(spikes > 0)
    spikes_frame[rows, cols, :] = [0, 255, 0]

  if polarity == DOWN_POLARITY or polarity == MERGED_POLARITY:
    rows, cols = numpy.where(spikes < 0)
    spikes_frame[rows, cols, :] = [0, 0, 255]

  return spikes_frame


@cython.boundscheck(False) # turn off bounds-checking for entire function
def render_comparison(numpy.ndarray[DTYPE_t, ndim=2] curr_frame,
                      numpy.ndarray[DTYPE_t, ndim=2] ref_frame,
                      numpy.ndarray[DTYPE_t, ndim=2] lap_curr,
                      numpy.ndarray[DTYPE_t, ndim=2] lap_ref,
                      numpy.ndarray[DTYPE_U8_t, ndim=3] spikes_frame,
                      DTYPE_t width, DTYPE_t height):
  """
    Compose a comparison of visual features
    -----------------------------------------------
    | current   |  reference  |   abs difference  |
    -----------------------------------------------
    | edges in  |  edges in   |   abs difference  |
    | current   |  reference  |   of edges        |
    -----------------------------------------------

  """
#~   cdef numpy.ndarray[DTYPE_U8_t, ndim=3] out = numpy.zeros([2*height, 4*width, 3], dtype=DTYPE_U8)
  cdef numpy.ndarray[DTYPE_U8_t, ndim=3] out = numpy.zeros([height, 4*width, 3], dtype=DTYPE_U8)

  out[:height, 0:width, 0] = curr_frame
  out[:height, 0:width, 1] = curr_frame
  out[:height, 0:width, 2] = curr_frame
  out[:height, width:2*width, 0] = ref_frame
  out[:height, width:2*width, 1] = ref_frame
  out[:height, width:2*width, 2] = ref_frame
#~   out[:, 2*width:3*width, 0] = (numpy.abs(curr_frame - ref_frame) > 0)*255
#~   out[:, 2*width:3*width, 1] = (numpy.abs(curr_frame - ref_frame) > 0)*255
#~   out[:, 2*width:3*width, 2] = (numpy.abs(curr_frame - ref_frame) > 0)*255
  out[:height, 2*width:3*width, 0] = numpy.abs(curr_frame - ref_frame)
  out[:height, 2*width:3*width, 1] = numpy.abs(curr_frame - ref_frame)
  out[:height, 2*width:3*width, 2] = numpy.abs(curr_frame - ref_frame)
  out[:height, 3*width:, :] = spikes_frame

#~   out[height:, 0:width, 0] = lap_curr
#~   out[height:, 0:width, 1] = lap_curr
#~   out[height:, 0:width, 2] = lap_curr
#~   out[height:, width:2*width, 0] = lap_ref
#~   out[height:, width:2*width, 1] = lap_ref
#~   out[height:, width:2*width, 2] = lap_ref
#~   out[height:, 2*width:3*width, 0] = numpy.abs(lap_curr - lap_ref)
#~   out[height:, 2*width:3*width, 1] = numpy.abs(lap_curr - lap_ref)
#~   out[height:, 2*width:3*width, 2] = numpy.abs(lap_curr - lap_ref)

  return out

@cython.boundscheck(False) # turn off bounds-checking for entire function
def split_spikes(numpy.ndarray[DTYPE_t, ndim=2] spikes,
                 numpy.ndarray[DTYPE_t, ndim=2] abs_diff,
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
  cdef numpy.ndarray[DTYPE_IDX_t, ndim=1] neg_rows, neg_cols, \
                                         pos_rows, pos_cols
  cdef numpy.ndarray[DTYPE_t, ndim=1]    neg_vals, pos_vals
  cdef DTYPE_t global_max = 0


  if polarity == RECTIFIED_POLARITY:
    # print("RECTIFIED_POLARITY")
    pos_rows, pos_cols = numpy.where(spikes != 0)
    pos_vals = abs_diff[pos_rows, pos_cols]
    neg_rows = neg_cols = numpy.array([], dtype=DTYPE_IDX)
    neg_vals = numpy.array([], dtype=DTYPE)
    if len(pos_vals) > 0:
      global_max = pos_vals.max()

  elif polarity == UP_POLARITY:
    pos_rows, pos_cols = numpy.where(spikes > 0)
    pos_vals = abs_diff[pos_rows, pos_cols]
    neg_rows = neg_cols = numpy.array([], dtype=DTYPE_IDX)
    neg_vals = numpy.array([], dtype=DTYPE)

    if len(pos_vals) > 0:
      global_max = pos_vals.max()


  elif polarity == DOWN_POLARITY:
    pos_rows, pos_cols = numpy.where(spikes < 0)
    pos_vals = abs_diff[pos_rows, pos_cols]
    neg_rows = neg_cols = numpy.array([], dtype=DTYPE_IDX)
    neg_vals = numpy.array([], dtype=DTYPE)

    if len(pos_vals) > 0:
      global_max = pos_vals.max()

  elif polarity == MERGED_POLARITY:
    neg_rows, neg_cols = numpy.where(spikes < 0)
    neg_vals = abs_diff[neg_rows, neg_cols]

    pos_rows, pos_cols = numpy.where(spikes > 0)
    pos_vals = abs_diff[pos_rows, pos_cols]

    if len(neg_vals) > 0 and len(pos_vals) > 0:
      global_max = max(neg_vals.max(), pos_vals.max())
    elif len(neg_vals) > 0:
      global_max = neg_vals.max()
    elif len(pos_vals) > 0:
      global_max = pos_vals.max()


  ####### ROWS, COLS, VALS
  return numpy.array([neg_rows, neg_cols, neg_vals], dtype=DTYPE), \
         numpy.array([pos_rows, pos_cols, pos_vals], dtype=DTYPE), \
         global_max

@cython.boundscheck(False) # turn off bounds-checking for entire function
cdef inline grab_spike_key(DTYPE_t row, DTYPE_t col,
                           DTYPE_U8_t flag_shift, DTYPE_U8_t data_shift,
                           DTYPE_U8_t data_mask,
                           DTYPE_U8_t is_pos_spike, 
                           DTYPE_U8_t key_coding=KEY_SPINNAKER):
  if key_coding == KEY_SPINNAKER:
    spike_key = spike_to_key(row, col, \
                             flag_shift, data_shift, data_mask,\
                             is_pos_spike)
  elif key_coding == KEY_XYP:
    spike_key = spike_to_xyp(row, col, \
                             is_pos_spike)
                             
  return spike_key

@cython.boundscheck(False) # turn off bounds-checking for entire function
cdef numpy.ndarray[DTYPE_t, ndim=1] spike_to_xyp(DTYPE_t row, DTYPE_t col,
                          DTYPE_U8_t is_pos_spike):
  return numpy.array([col, row, is_pos_spike])


@cython.boundscheck(False) # turn off bounds-checking for entire function
cdef DTYPE_t spike_to_key(DTYPE_t row, DTYPE_t col,
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

      The output format is: [up|down][row][col]
    """
    
    cdef DTYPE_U16_t d = 0

    #up/down bit
    if not is_pos_spike:
        d = d | 1 << flag_shift

    # row bits
    d = d | (row & data_mask) <<  data_shift
    # col bits
    d = d | (col & data_mask)

    return d



@cython.boundscheck(False) # turn off bounds-checking for entire function
def make_spike_lists_rate(numpy.ndarray[DTYPE_t, ndim=2] pos_spikes,
                          numpy.ndarray[DTYPE_t, ndim=2] neg_spikes,
                          DTYPE_t global_max,
                          DTYPE_t threshold,
                          DTYPE_U8_t flag_shift,
                          DTYPE_U8_t data_shift,
                          DTYPE_U8_t data_mask,
                          DTYPE_t max_time_ms,
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
  cdef unsigned int max_spikes = max_time_ms, \
                    len_neg = len(neg_spikes[0]), \
                    len_pos = len(pos_spikes[0])
  cdef unsigned int max_pix = len_neg + len_pos
  cdef unsigned int list_idx

  cdef Py_ssize_t spike_idx, pix_idx, neg_idx
  cdef DTYPE_t spike_key
  cdef list list_of_lists = list()
  cdef DTYPE_t val

  for list_idx in range(max_spikes):
    list_of_lists.append( list() )

  for pix_idx in range(max_pix):
    spike_key = 0

    if pix_idx < len_pos:
      spike_key = grab_spike_key(pos_spikes[ROWS, pix_idx], \
                                   pos_spikes[COLS, pix_idx], \
                                   flag_shift, data_shift, data_mask,\
                                   is_pos_spike = 1,
                                   key_coding=key_coding)
          
          
      val = pos_spikes[VALS, pix_idx]/threshold
      spike_idx = min(max_spikes-1, val)

    else:
      neg_idx = pix_idx - len_pos
      spike_key = grab_spike_key(neg_spikes[ROWS, neg_idx], \
                                 neg_spikes[COLS, neg_idx], \
                                 flag_shift, data_shift, data_mask,\
                                 is_pos_spike = 0,
                                 key_coding=key_coding)

      val = neg_spikes[VALS, neg_idx]/threshold
#~       print("neg rate spikes val, key", val, spike_key)
      spike_idx = min(max_spikes-1, val)
      
    for list_idx in range(spike_idx):
      list_of_lists[list_idx].append(spike_key)



  return list_of_lists



@cython.boundscheck(False) # turn off bounds-checking for entire function
def make_spike_lists_time(numpy.ndarray[DTYPE_t, ndim=2] pos_spikes,
                          numpy.ndarray[DTYPE_t, ndim=2] neg_spikes,
                          DTYPE_t global_max,
                          DTYPE_U8_t flag_shift,
                          DTYPE_U8_t data_shift,
                          DTYPE_U8_t data_mask,
                          DTYPE_t num_bins,
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
  cdef int num_thresh = 0


  cdef Py_ssize_t time_idx, pix_idx, neg_idx
  cdef DTYPE_t spike_key
  cdef list list_of_lists = list()

  for time_idx in range(num_bins):
    list_of_lists.append( list() )

#~   print list_of_lists

#~   for pix_idx in prange(max_pix, nogil=True):
  for pix_idx in range(max_pix):
    if pix_idx < len_pos:
      spike_key = grab_spike_key(pos_spikes[ROWS, pix_idx], \
                                 pos_spikes[COLS, pix_idx], \
                                 flag_shift, data_shift, data_mask,\
                                 is_pos_spike = 1,
                                 key_coding=key_coding)
      num_thresh = min(pos_spikes[VALS, pix_idx]/min_threshold  - 1, num_bins - 1)
    else:
      neg_idx = pix_idx - len_pos
      spike_key = grab_spike_key(neg_spikes[ROWS, neg_idx], \
                                 neg_spikes[COLS, neg_idx], \
                                 flag_shift, data_shift, data_mask,\
                                 is_pos_spike = 0,
                                 key_coding=key_coding)

      num_thresh = max( min(neg_spikes[VALS, neg_idx]/min_threshold - 1, 
                            num_bins - 1),
                        0 )

    time_idx = num_thresh # num_bins - num_thresh - 1
#~     print "num_bins(%s), num_thresh (%s), time_idx (%s)"%(num_bins, num_thresh, time_idx)
    list_of_lists[time_idx].append( spike_key )

  return list_of_lists



@cython.boundscheck(False) # turn off bounds-checking for entire function
def make_spike_lists_time_bin(numpy.ndarray[DTYPE_t, ndim=2] pos_spikes,
                              numpy.ndarray[DTYPE_t, ndim=2] neg_spikes,
                              DTYPE_t global_max,
                              DTYPE_U8_t flag_shift,
                              DTYPE_U8_t data_shift,
                              DTYPE_U8_t data_mask,
                              DTYPE_t max_time_ms,
                              DTYPE_t min_threshold,
                              DTYPE_t max_threshold,
                              DTYPE_t num_bins,
                              numpy.ndarray[DTYPE_U8_t, ndim=1] log2_table,
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

  cdef numpy.ndarray[DTYPE_IDX_t, ndim=1] indices
  cdef Py_ssize_t time_idx, pix_idx, neg_idx
  cdef DTYPE_t spike_key
  cdef list list_of_lists = list()
  cdef DTYPE_U8_t byte_code

  #8-bit images
  for spike_key in range(num_bins):
    list_of_lists.append(list())

#~   for pix_idx in prange(max_pix, nogil=True):
  for pix_idx in range(max_pix):
    if pix_idx < len_pos:
      spike_key = grab_spike_key(pos_spikes[ROWS, pix_idx], \
                                 pos_spikes[COLS, pix_idx], \
                                 flag_shift, data_shift, data_mask,\
                                 is_pos_spike = 1,
                                 key_coding=key_coding)

      byte_code = log2_table[ pos_spikes[VALS, pix_idx] ]

      indices, = numpy.where( numpy.unpackbits(numpy.uint8(byte_code)) )

      for i in indices:
        list_of_lists[i].append(spike_key)

    else:
      neg_idx = pix_idx - len_pos
      spike_key = grab_spike_key(neg_spikes[ROWS, neg_idx], \
                                 neg_spikes[COLS, neg_idx], \
                                 flag_shift, data_shift, data_mask,\
                                 is_pos_spike = 0,
                                 key_coding=key_coding)

      byte_code = log2_table[ neg_spikes[VALS, neg_idx] ]

      indices, = numpy.where( numpy.unpackbits(numpy.uint8(byte_code)) )

      for i in indices:
        list_of_lists[i].append(spike_key)


  return list_of_lists



@cython.boundscheck(False) # turn off bounds-checking for entire function
def make_spike_lists_time_bin_thr(numpy.ndarray[DTYPE_t, ndim=2] pos_spikes,
                                  numpy.ndarray[DTYPE_t, ndim=2] neg_spikes,
                                  DTYPE_t global_max,
                                  DTYPE_U8_t flag_shift,
                                  DTYPE_U8_t data_shift,
                                  DTYPE_U8_t data_mask,
                                  DTYPE_t max_time_ms,
                                  DTYPE_t min_threshold,
                                  DTYPE_t max_threshold,
                                  DTYPE_t num_bins,
                                  numpy.ndarray[DTYPE_U8_t, ndim=1] log2_table,
                                  DTYPE_U8_t key_coding=KEY_SPINNAKER):
  """
    Convert spike (row, col, val, sign) lists into a list of Address
    Event Representation (AER) encoded spikes. Time/binary-encoded number of
    thresholds rebased by the difference.
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

  cdef numpy.ndarray[DTYPE_IDX_t, ndim=1] indices
  cdef Py_ssize_t time_idx, pix_idx, neg_idx
  cdef DTYPE_t spike_key
  cdef list list_of_lists = list()
  cdef unsigned char byte_code

  for spike_key in range(num_bins):
    list_of_lists.append(list())

#~   for pix_idx in prange(max_pix, nogil=True):
  for pix_idx in range(max_pix):
    if pix_idx < len_pos:
      spike_key = grab_spike_key(pos_spikes[ROWS, pix_idx], \
                                 pos_spikes[COLS, pix_idx], \
                                 flag_shift, data_shift, data_mask,\
                                 is_pos_spike = 1,
                                 key_coding=key_coding)

      byte_code = log2_table[ pos_spikes[VALS, pix_idx]/min_threshold ]

      indices, = numpy.where( numpy.unpackbits(numpy.uint8(byte_code)) )
#~       print "byte_val (%s)"%byte_code
#~       print indices

      for i in indices:
        if i > num_bins:
          i = num_bins
        list_of_lists[num_bins - i].append(spike_key)

    else:
      neg_idx = pix_idx - len_pos
      spike_key = grab_spike_key(neg_spikes[ROWS, neg_idx], \
                                 neg_spikes[COLS, neg_idx], \
                                 flag_shift, data_shift, data_mask,\
                                 is_pos_spike = 0,
                                 key_coding=key_coding)

      byte_code = log2_table[ neg_spikes[VALS, neg_idx]/min_threshold ]

      indices, = numpy.where( numpy.unpackbits(numpy.uint8(byte_code)) )
#~       print "byte_val (%s)"%byte_code
#~       print indices

      for i in indices:
        if i > num_bins:
          i = num_bins

        list_of_lists[num_bins - i].append(spike_key)

  return list_of_lists



def generate_log2_table(max_active_bits, bit_resolution):
  """Create a look-up table for the possible values in the range (0, 2^bit_resolution)
    one table per active bits in the range (0, max_active_bits]
  """
  cdef numpy.ndarray[DTYPE_U8_t, ndim=2] log2_table = numpy.zeros((max_active_bits, 2**bit_resolution), dtype=DTYPE_U8)
  cdef int active_bits, value

  for active_bits in range(max_active_bits):
    for value in range( 2**bit_resolution ):

      log2_table[active_bits][value] = encode_time_n_bits_single(value, active_bits)


  return log2_table

#######################################################################################

def mask_image(numpy.ndarray[DTYPE_t, ndim=2] original,
               numpy.ndarray[DTYPE_FLOAT_t, ndim=2] mask):
    
    return DTYPE(original*mask)


def traverse_image(numpy.ndarray[DTYPE_t, ndim=2] original,
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
  cdef numpy.ndarray[DTYPE_t, ndim=2] moved = bg_gray*numpy.ones_like(original, dtype=DTYPE)
  cdef int width = len(original[0])
  cdef int n = DTYPE(frame_number*speed)

  if n == 0:
    n = 1

  if n <= width:
    moved[:, :n]  = original[:, -n:]
  elif n < 2*width:
    n = 2*width - n + 1
    moved[:, -n:] = original[:, :n]

  return moved



def fade_image(numpy.ndarray[DTYPE_t, ndim=2] original,
               DTYPE_t frame_number, DTYPE_t half_frame,
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
  cdef numpy.ndarray[DTYPE_t, ndim=2] moved = bg_gray*numpy.ones_like(original, dtype=DTYPE)

  cdef DTYPE_FLOAT_t alpha = 0.0
  if frame_number < half_frame - 1:
    alpha = DTYPE_FLOAT(frame_number)/DTYPE_FLOAT(half_frame)
  elif frame_number > half_frame + 1:
    alpha = DTYPE_FLOAT((half_frame*2) - frame_number)/DTYPE_FLOAT(half_frame)
  else:
    alpha = 1.0

  moved[:] = DTYPE(original*alpha)

  return moved

cdef move_image(numpy.ndarray[DTYPE_t, ndim=2] original,
                DTYPE_t delta_x,
                DTYPE_t delta_y,
                DTYPE_t bg_gray):
  cdef numpy.ndarray[DTYPE_t, ndim=2] moved = bg_gray*numpy.ones_like(original, dtype=DTYPE)
  cdef DTYPE_t new_x0, new_x1, old_x0, old_x1
  cdef DTYPE_t new_y0, new_y1, old_y0, old_y1
  cdef DTYPE_t width  = len(original[0])
  cdef DTYPE_t height = len(original)

  if delta_x < 0:
    new_x0 = 0;             new_x1 = delta_x
    old_x0 = abs(delta_x);  old_x1 = width
  elif delta_x > 0:
    new_x0 = delta_x;  new_x1 = width
    old_x0 = 0;        old_x1 = -delta_x
  else:
    new_x0 = 0;        new_x1 = width
    old_x0 = 0;        old_x1 = width

  if delta_y > 0:
    new_y0 = 0;        new_y1 = -delta_y
    old_y0 = delta_y;  old_y1 = height
  elif delta_y < 0:
    new_y0 = abs(delta_y);  new_y1 = height
    old_y0 = 0;             old_y1 = delta_y
  else:
    new_y0 = 0;             new_y1 = height
    old_y0 = 0;             old_y1 = height

  moved[new_y0:new_y1, new_x0:new_x1] = original[old_y0:old_y1, old_x0:old_x1]
  return moved


def usaccade_image(numpy.ndarray[DTYPE_t, ndim=2] original,
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
  if frame_number%frames_per_usaccade != 0:
    return move_image(original, center_x, center_y, bg_gray), center_x, center_y

  numpy.random.seed(seed=numpy.uint32(time.time()*1000000))
  center_x += numpy.random.randint(-max_delta, max_delta+1)
  center_y += numpy.random.randint(-max_delta, max_delta+1)

  return move_image(original, center_x, center_y, bg_gray), center_x, center_y



def attention_image(numpy.ndarray[DTYPE_t, ndim=2] original,
                    numpy.ndarray[DTYPE_t, ndim=2] previous,
                    numpy.ndarray[DTYPE_t, ndim=2] reference,
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

  #not the right frame, do micro-saccade
  if frame_number%frame_saccade != 0:
    return usaccade_image(original, frame_number, frames_per_usaccade, \
                          max_delta, center_x, center_y, bg_gray)


  # simulate saccade
  cdef numpy.ndarray[DTYPE_t, ndim=2] moved = bg_gray*numpy.ones_like(original, dtype=DTYPE)
  cdef int new_w = len(original[0])/2
  cdef numpy.ndarray[DTYPE_t, ndim=2] tiny_prev = cv2.resize(previous, (new_w, new_w), \
                                                             interpolation=CV_INTER_AREA).astype(DTYPE)

  cdef numpy.ndarray[DTYPE_t, ndim=2] tiny_ref  = cv2.resize(reference, (new_w, new_w), \
                                                             interpolation=CV_INTER_AREA).astype(DTYPE)
  cdef numpy.ndarray[DTYPE_t, ndim=2] diff = numpy.abs(tiny_prev - tiny_ref).astype(DTYPE)

  cdef numpy.ndarray[DTYPE_IDX_t, ndim=1] top_n = numpy.argsort(diff.reshape(new_w*new_w))\
                                                       .astype(DTYPE_IDX)[-5:]

  cdef int max_idx = top_n[numpy.random.randint(5)]
  cdef int row = (max_idx/new_w)*2
  cdef int col = (max_idx%new_w)*2

  center_x = new_w - col
  center_y = new_w - row
  if center_x > new_w or center_x < -new_w or\
     center_y > new_w or center_y < -new_w:
       center_x = 0
       center_y = 0


  return move_image(original, center_x, center_y, bg_gray), center_x, center_y

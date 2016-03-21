from __future__ import print_function
import time

import numpy
from numpy import int16, uint16, uint8, float16, log2

import cv2
from cv2 import cvtColor as convertColor, COLOR_BGR2GRAY, COLOR_GRAY2RGB,\
                resize

try:                  #nearest neighboor interpolation
  from cv2.cv import CV_INTER_NN, \
                     CV_CAP_PROP_FRAME_WIDTH, \
                     CV_CAP_PROP_FRAME_HEIGHT, \
                     CV_CAP_PROP_FPS
except:
  from cv2 import INTER_NEAREST as CV_INTER_NN, \
                  CAP_PROP_FRAME_WIDTH as CV_CAP_PROP_FRAME_WIDTH, \
                  CAP_PROP_FRAME_HEIGHT as CV_CAP_PROP_FRAME_HEIGHT, \
                  CAP_PROP_FPS as CV_CAP_PROP_FPS

import pyximport; pyximport.install()
from pydvs.generate_spikes import *

MODE_128 = "128"
MODE_64  = "64"
MODE_32  = "32"
MODE_16  = "16"

UP_POLARITY     = "UP"
DOWN_POLARITY   = "DOWN"
MERGED_POLARITY = "MERGED"
POLARITY_DICT   = {UP_POLARITY: uint8(0), 
                 DOWN_POLARITY: uint8(1), 
                 MERGED_POLARITY: uint8(2),
                 0: UP_POLARITY,
                 1: DOWN_POLARITY,
                 2: MERGED_POLARITY}

OUTPUT_RATE         = "RATE"
OUTPUT_TIME         = "TIME"
OUTPUT_TIME_BIN     = "TIME_BIN"
OUTPUT_TIME_BIN_THR = "TIME_BIN_THR"

BEHAVE_MICROSACCADE = "SACCADE"
BEHAVE_ATTENTION    = "ATTENTION"
BEHAVE_TRAVERSE     = "TRAVERSE"
BEHAVE_FADE         = "FADE"

IMAGE_TYPES = ["png", 'jpeg', 'jpg']


# -------------------------------------------------------------------- #
# grab / rescale frame                                                 #

def grab_first(dev, res):
  valid, raw = dev.read()
  height, width, depth = raw.shape
  new_height = res
  new_width = int( float(new_height*width)/float(height) )
  col_from = (new_width - res)//2
  col_to   = col_from + res
  img = resize(convertColor(raw, COLOR_BGR2GRAY).astype(int16),
               (new_width, new_height), interpolation=CV_INTER_NN)[:, col_from:col_to]

  return img, new_width, new_height, col_from, col_to

def grab_frame(dev, width, height, col_from, col_to):
  valid, raw = dev.read()
  img = resize(convertColor(raw, COLOR_BGR2GRAY).astype(int16),
               (width, height), interpolation=CV_INTER_NN)[:, col_from:col_to]

  return img

#----------------------------------------------------------------------#

mode = MODE_128
cam_res = int(mode)
width = cam_res # square output
height = cam_res
shape = (height, width)
#cam_res = 256 # <- can be done, but spynnaker doesn't suppor such resolution

data_shift = uint8( log2(cam_res) )
up_down_shift = uint8(2*data_shift)
data_mask = uint8(cam_res - 1)

polarity = POLARITY_DICT[ MERGED_POLARITY ]
output_type = OUTPUT_TIME
history_weight = 1.0
threshold = 12 # ~ 0.05*255
max_threshold = 180 # 12*15 ~ 0.7*255

scale_width = 0
scale_height = 0
col_from = 0
col_to = 0

curr     = numpy.zeros(shape,     dtype=int16) 
ref      = 128*numpy.ones(shape,  dtype=int16) 
spikes   = numpy.zeros(shape,     dtype=int16) 
diff     = numpy.zeros(shape,     dtype=int16) 
abs_diff = numpy.zeros(shape,     dtype=int16) 

# just to see things in a window
spk_img  = numpy.zeros((height, width, 3), uint8)

num_bits = 6   # how many bits are used to represent exceeded thresholds
num_active_bits = 2 # how many of bits are active
log2_table = generate_log2_table(num_active_bits, num_bits)[num_active_bits - 1]
spike_lists = None
pos_spks = None
neg_spks = None
max_diff = 0


# -------------------------------------------------------------------- #
# inhibition related                                                   #

inh_width = 2
is_inh_on = True
inh_coords = generate_inh_coords(width, height, inh_width)


# -------------------------------------------------------------------- #
# camera/frequency related                                             #

video_dev = cv2.VideoCapture(0) # webcam
#video_dev = cv2.VideoCapture('/path/to/video/file') # webcam

#ps3 eyetoy can do 125fps
try:
  video_dev.set(CV_CAP_PROP_FPS, 125)
except:
  pass
  
fps = video_dev.get(CV_CAP_PROP_FPS)
max_time_ms = int(1000./fps)


#---------------------- main loop -------------------------------------#

is_first_pass = True
start_time = time.time()
end_time = 0
frame_count = 0
while(True):
  # get an image from video source
  if is_first_pass:
    curr[:], scale_width, scale_height, col_from, col_to = grab_first(video_dev, cam_res)
  else:
    curr[:] = grab_frame(video_dev, scale_width,  scale_height, col_from, col_to)
  
  # do the difference
  diff[:], abs_diff[:], spikes[:] = thresholded_difference(curr, ref, threshold)
  
  # inhibition ( optional ) 
  if is_inh_on:
    spikes[:] = local_inhibition(spikes, abs_diff, inh_coords, 
                                 width, height, inh_width)
  
  # update the reference
  ref[:] = update_reference_time_binary_thresh(abs_diff, spikes, ref,
                                               threshold, max_time_ms,
                                               num_active_bits,
                                               history_weight,
                                               log2_table)
  
  # convert into a set of packages to send out
  neg_spks, pos_spks, max_diff = split_spikes(spikes, abs_diff, polarity)
  
  # this takes too long, could be parallelized at expense of memory
  spike_lists = make_spike_lists_time_bin_thr(pos_spks, neg_spks,
                                              max_diff,
                                              up_down_shift, data_shift, data_mask,
                                              max_time_ms,
                                              threshold, 
                                              max_threshold,
                                              num_bits,
                                              log2_table)
  
  spk_img[:] = render_frame(spikes, curr, cam_res, cam_res, polarity)
  cv2.imshow ("spikes", spk_img.astype(uint8))  
  
  if cv2.waitKey(1) & 0xFF == ord('q'):
    break
  
  end_time = time.time()
  
  if end_time - start_time >= 1.0:
    print("%d frames per second"%(frame_count))
    frame_count = 0
    start_time = time.time()
  else:
    frame_count += 1


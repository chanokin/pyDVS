from __future__ import print_function

import cv2
import cv
import numpy
import sys

### If OpenCL is available, then use that ###
### otherwise use Numpy for faster math   ###
from numpy_dvs import NumpyDVS
try:
  import pyopencl as cl
  using_opencl_dvs = True
  from opencl_dvs import OpenCL_DVS
except ImportError:
  using_opencl_dvs = False
  print("Using Numpy backend")



class pyDVS():
  
  
  def __init__(self, video_capture_id, gray_bin_size = 10, 
               threshold=0.2, threshold_rate=0.1,
               max_threshold=64, min_threshold=0,
               starting_val=50, input_to_log=False,
               use_conv=False,
               force_numpy=True):

    self.gray_bin_size = gray_bin_size #divide 255 levels into bins
    self.current_frame  = None
    self.result_indices = None
    self.result_frame = None
    #~ self.threshold = int(255*threshold)
    self.threshold = threshold
    self.threshold_rate = int(255*threshold_rate)
    self.max_threshold = max_threshold if max_threshold > self.threshold else self.threshold
    self.min_threshold = min_threshold
    self.frame_width    = 0
    self.frame_height   = 0
    self.frame_size     = 0
    self.frames_per_second = 0
    self.starting_val = 0
    self.first_frame = None
    self.input_to_log = input_to_log
    self.use_conv = use_conv
    self.capture_device = cv2.VideoCapture(video_capture_id)
    
    assert self.capture_device.isOpened(), \
           'Unable to open VideoCapture device!'
    
    self.assess_video_features()
    
    if force_numpy or not using_opencl_dvs:
      print("Using Numpy backend")
      self.processor = NumpyDVS(self.frame_width, self.frame_height, 
                                self.threshold, self.threshold_rate,
                                self.max_threshold, self.min_threshold,
                                self.starting_val, self.input_to_log,
                                self.use_conv)
                                
    else:
      print("Using OpenCL backend")

      self.processor = OpenCL_DVS(self.frame_width, self.frame_height, 
                                  self.threshold, self.threshold_rate,
                                  self.max_threshold, self.min_threshold,
                                  self.starting_val, self.input_to_log,
                                  self.use_conv)
      

    
    self.current_frame  = numpy.zeros(self.frame_size,  dtype=numpy.uint8)
    self.result_frame   = numpy.zeros(self.frame_size,  dtype=numpy.int16)
    self.result_indices = numpy.zeros(self.frame_size,  dtype=numpy.int16)

    self.time_bin_size = self.gray_bin_size/(self.frames_per_second*(2.**8. - 1.))
  
  
  def assess_video_features(self):
    self.frame_width  = self.capture_device.get(cv2.cv.CV_CAP_PROP_FRAME_WIDTH)
    self.frame_height = self.capture_device.get(cv2.cv.CV_CAP_PROP_FRAME_HEIGHT)

    if self.frame_width == 0:
      ret, self.first_frame = self.capture_device.read()
      self.frame_width = self.first_frame.shape[1]
      self.frame_height = self.first_frame.shape[0]

    self.frame_size   = self.frame_height*self.frame_width    
    
    assert self.frame_width > 0,  "Frame width <= 0"
    assert self.frame_height > 0, "Frame height <= 0"
    
    self.frames_per_second = self.capture_device.get(cv2.cv.CV_CAP_PROP_FPS)
    if self.frames_per_second == 0:
      self.frames_per_second = 24.
    
    
  def __del__(self):
    self.capture_device.release()
    cv2.destroyAllWindows()


  def update(self):
    
    ret, frame = self.capture_device.read()
    
    if ret is False:
      print("WARNING: Failed to read from capture device", file=sys.stderr)
    else:
      numpy.copyto(self.current_frame, 
                   cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).reshape(self.frame_size))
      img, ind = self.processor.process_frame(self.current_frame)
      numpy.copyto(self.result_frame,   img)
      numpy.copyto(self.result_indices, ind)
    
    return ret
    
  
  def emit(self, emit_func, time_base=0.):
    if emit_func is None:
      return
      
    current_pack = []
    pos_vals_over = False
    neg_vals_over = False
    count_start_pos = 0
    count_start_neg = 0
    pack_count = 0
    for g_begin in xrange((2**8 - 1, 0, -self.gray_bin_size)):
      del current_pack[:]
      time_stamp = time_base + pack_count*self.time_bin_size
      
      if not pos_vals_over:
        for idx in xrange(count_start_pos, self.frame_size):
          img_idx = self.result_indices[idx]
          val = self.result_frame[img_idx]
          
          if val < 1:
            pos_vals_over = True
            
          if val < g_begin - self.gray_bin_size:
            break
          count_start_pos += 1
          current_pack.append((img_idx, time_stamp, 1))
      
      if not neg_vals_over:
        for idx in xrange(count_start_neg, self.frame_size):
          img_idx = self.result_indices[-idx]
          val = self.result_frame[img_idx]

          if val > -1:
            neg_vals_over = True

          val = numpy.abs(val)
          
          if val < g_begin - self.gray_bin_size:
            break
          
          count_start_neg += 1
          current_pack.append((img_idx, time_stamp, -1))
      
      emmit_func(current_pack)
      pack_count += 1
  
  
  def run(self, emit_func):
    start_time = time.time()
    update_possible = self.update()
    end_time = time.time()
    if update_possible is False:
      raise Exception("Unable to grab a new frame from video device")
    self.emit(emit_func, time_base=start_time-end_time)


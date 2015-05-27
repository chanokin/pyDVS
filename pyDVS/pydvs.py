import cv2
import cv
import numpy

### If OpenCL is available, then use that ###
### otherwise use Numpy for faster math   ###
try:
  import opencl as cl
  using_opencl_dvs = True
  from opencl_dvs import OpenCL_DVS as frame_processor
except ImportError:
  using_opencl_dvs = False
  from numpy_dvs import NumpyDVS as frame_processor



class CamDVS():
  
  def __init__(self, video_capture_id):
    self.current_frame  = None
    self.previous_frame = None
    self.output_frame   = None
    self.frame_width    = 0
    self.frame_height   = 0
    self.capture_device = cv2.VideoCapture(video_capture_id)
    
    assert self.capture_device.isOpened(), 
           'Unable to open VideoCapture device!'
  
    self.processor = frame_processor(image_width, image_height)
    self.assess_video_features()
    self.current_frame = numpy.zeros((self.frame_height, self.frame_width), \
                                     dtype=numpy.int8)
  
  def assess_video_features(self):
    self.frame_width  = self.capture_device.get(self.__CV_CAP_WIDTH_ID)
    self.frame_height = self.capture_device.get(self.__CV_CAP_HEIGHT_ID)
    
    assert self.frame_width > 0,  "Frame width <= 0"
    assert self.frame_height > 0, "Frame height <= 0"

  def __del__(self):
    self.capture_device.release()
    cv2.destroyAllWindows()


  def update(self):
    ret, frame = cap.read()
    self.previous_frame = self.current_frame.copy()
    self.current_frame  = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    self.output_frame   = self.processor.process_frame(self.current_frame, \
                                                       self.previous_frame)
  
  
  def emit(self):



  def run(self):


  __CV_CAP_HEIGHT_ID = 4
  __CV_CAP_WIDTH_ID  = 3

import numpy

class NumpyDVS():
  
  def __init__(self, image_width, image_height):
    pass
  
  def process_frame(self, current_frame, previous_frame):
    ii8 = numpy.iinfo(numpy.int8) 
    diff = current_frame.astype(numpy.int16) - previous_frame.astype(numpy.int16)
    
    return numpy.clip(diff, ii8.min, ii8.max).astype(numpy.int8)

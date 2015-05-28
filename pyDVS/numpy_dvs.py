import numpy

class NumpyDVS():
  
  def __init__(self, image_width, image_height):
    pass
  
  def process_frame(self, current_frame, previous_frame):
    diff = current_frame.astype(numpy.int16) - previous_frame.astype(numpy.int16)
    pos = diff*(diff > 0)
    neg = numpy.abs(diff)*(diff < 0)
    
    return pos, neg

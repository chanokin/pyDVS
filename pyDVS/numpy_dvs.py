import numpy

class NumpyDVS():
  
  def __init__(self, image_width, image_height):
    pass
  
  def process_frame(self, current_frame, previous_frame):
    return ( (current_frame > previous_frame) + \
             (current_frame < previous_frame)*(-1) ).astype(numpy.int8)

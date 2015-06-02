import numpy

class NumpyDVS():
  
  def __init__(self, image_width, image_height, threshold):
    self.threshold = threshold

  
  def process_frame(self, current_frame, previous_frame):
    '''Returns difference value and sorted indices for further processing'''
    diff = current_frame.astype(numpy.int16) - previous_frame.astype(numpy.int16)
    diff = diff*(diff*diff > self.threshold*self.threshold)
    indices = numpy.argsort(diff)

    return diff, indices

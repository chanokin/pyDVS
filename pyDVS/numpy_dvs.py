import numpy

class NumpyDVS():
  
  def __init__(self, image_width, image_height, 
               threshold=0.01, threshold_rate=0.01, max_threshold=64,
               min_threshold=0, starting_val=50, input_to_log=False,
               use_conv=False):
    self.threshold_rate = threshold_rate
    self.threshold = numpy.ones(image_height*image_width, dtype=numpy.uint16)*threshold
    self.previous_frame = numpy.ones(image_height*image_width, dtype=numpy.int16)*127
    self.max_threshold = max_threshold
    self.min_threshold = min_threshold
    
  def process_frame(self, current_frame):
    '''Returns difference value and sorted indices for further processing'''
    diff = current_frame.astype(numpy.int16) - self.previous_frame
    spiked_mask = diff*diff > self.threshold*self.threshold
    diff = diff*spiked_mask
    
    spiked_mask *= 2
    spiked_mask -= 1

    self.threshold += spiked_mask*self.threshold_rate
    numpy.copyto(self.threshold, 
                 numpy.clip(self.threshold, self.min_threshold, self.max_threshold))
    
    
    indices = numpy.argsort(diff)

    numpy.copyto(self.previous_frame, current_frame)
    return diff, indices

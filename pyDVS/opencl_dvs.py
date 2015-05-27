import pyopencl as cl
import pyopencl.array as cl_array

class OpenCL_DVS():
  
  def __init__(self, image_width, image_height):
    self.current_frame_gpu  = None
    self.previous_frame_gpu = None
    self.result_gpu = None
    

  
  def process_frame(self, current_frame, previous_frame):
    return ( (current_frame > previous_frame) + \
             (current_frame < previous_frame)*(-1) ).astype(numpy.int8)


  def create_context(self):
    platforms = cl.get_platforms()
    if len(platforms) == 0:
        print "Failed to find any OpenCL platforms."
        return None

    devices = platforms[0].get_devices(cl.device_type.GPU)
    if len(devices) == 0:
        print "Could not find GPU device, trying CPU..."
        devices = platforms[0].get_devices(cl.device_type.CPU)
        if len(devices) == 0:
            print "Could not find OpenCL GPU or CPU device."
            return None

    context = cl.Context([devices[0]])
    return context, devices[0]

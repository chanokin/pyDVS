import pyopencl as cl
import pyopencl.array as cl_array
from pyopencl.algorithm import RadixSort
import numpy
import sys
import os
os.environ['PYOPENCL_COMPILER_OUTPUT'] = '1'

    
class OpenCL_DVS():
  
  def __init__(self, image_width, image_height, threshold, cache_dir="./ocl_kernel_cache"):
    self.cache_dir = cache_dir
    self.image_width  = image_width
    self.image_height = image_height
    self.threshold = threshold
    
    self.workgroup_shape = (8, 8)
    self.global_work_shape = (RoundUp(self.workgroup_shape[0], self.image_width),
                              RoundUp(self.workgroup_shape[1], self.image_height))

    self.array_size = self.image_width*self.image_height
    
    self.context, self.device = self.create_context()

    if self.context == None:
      print "Failed to create OpenCL context."
      return 1
    
    self.queue = cl.CommandQueue(self.context, self.device)

    self.current_frame_gpu  = cl_array.zeros(self.queue, self.array_size,\
                                             numpy.uint8)
    self.previous_frame_gpu = cl_array.zeros(self.queue, self.array_size,\
                                             numpy.uint8)
    self.result_gpu  = cl_array.zeros(self.queue, self.array_size, numpy.int16)
    
    self.program = self.init_program()
  

  def init_program(self):
    prog_src = self.kernel_str()
    program = cl.Program(self.context, prog_src)
    options = ['-Werror', 
               #'-cl-mad-enable', 
               '-cl-single-precision-constant', 
               '-cl-no-signed-zeros', 
               #'-cl-fast-relaxed-math',
               ]

    if sys.platform == "darwin":
      print("Mac OS X!")
      options.append('-I /System/Library/Frameworks/OpenCL.framework/Versions/A/Headers/')
      options.append('-I /System/Library/Frameworks/OpenCL.framework/Versions/Current/lib/clang/3.2/include/')
      options.append('-I /System/Library/Frameworks/OpenCL.framework/Versions/Current/lib/clang/2.0/include/')
      options.append('-I /System/Library/Frameworks/OpenCL.framework/Versions/Current/Libraries/')
      options.append('-I /System/Library/Frameworks/OpenCL.framework/Versions/Current/Headers/')

    program.build(options=options, devices=[self.device], cache_dir=self.cache_dir)

    return program
    

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


  def kernel_str(self):
    kernel = """
#pragma OPENCL EXTENSION cl_amd_media_ops : enable
#pragma OPENCL EXTENSION cl_khr_byte_addressable_store : enable

__kernel void difference(__global uchar* current, __global uchar* previous,
                         __global short*  output){
  uint col = get_global_id(0), row = get_global_id(1),
       global_idx = row*IMAGE_WIDTH + col;
       
  int curr, prev, diff;

  if(global_idx < IMAGE_SIZE){
    curr = current[global_idx];
    prev = previous[global_idx];
    diff = curr - prev;
    if( diff*diff > THRESHOLD ){
      output[global_idx] = diff;
    }
    else{
      output[global_idx] = 0;
    }
  }
}
"""
    kernel = kernel.replace("IMAGE_SIZE", str(self.image_height*self.image_width))
    kernel = kernel.replace("IMAGE_WIDTH", str(self.image_width))
    kernel = kernel.replace("THRESHOLD", str(self.threshold*self.threshold))
    
    return kernel


  def process_frame(self, current_frame, previous_frame):
    '''Returns difference value and sorted indices for further processing'''
    
    self.current_frame_gpu  = cl_array.to_device(self.queue, current_frame)
    self.previous_frame_gpu = cl_array.to_device(self.queue, previous_frame)
    self.queue.finish()
    self.program.difference(self.queue, self.global_work_shape, self.workgroup_shape,
                            self.current_frame_gpu.data, self.previous_frame_gpu.data,
                            self.result_gpu.data)
    self.queue.finish()
    diff = self.result_gpu.get()
    
    indices = numpy.argsort(diff)
    
    return diff, indices



#######################################################################
def RoundUp(groupSize, globalSize):

  if  globalSize <= groupSize :
    return groupSize

  r = globalSize % groupSize
  if r == 0:
    return globalSize
  else:
    return globalSize + groupSize - r

import pyopencl as cl
import pyopencl.array as cl_array


    
class OpenCL_DVS():
  
  def __init__(self, image_width, image_height):

    self.image_width  = image_width
    self.image_height = image_height
      
    self.workgroup_shape = (8, 8)
    self.global_work_shape = (RoundUp(workgroup_shape[0], image_height),
                              RoundUp(workgroup_shape[1], image_width))
    self.array_size = image_width*image_height
    
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
    kernel = "#pragma OPENCL EXTENSION cl_amd_media_ops : enable\n \
#pragma OPENCL EXTENSION cl_khr_byte_addressable_store : enable \n \
\n \
__kernel void difference(__global uchar* current, __global uchar* previous,\n \
                         __global short*  output){\n \
  uint col = get_global_id(0), row = get_global_id(1),\n \
             global_idx = row*IMAGE_WIDTH + col;\n \
  uint curr, prev, diff;\n\
\n\
  if(global_idx < IMAGE_SIZE){\n\
    curr = current[global_idx];\n\
    prev = previous[global_idx];\n\
    diff = curr - prev;\n\
    if( curr < prev ){\n\
      output[global_idx] = diff < -128? -128: diff;\n\
    }\n\
    else if( prev < curr ){\n\
      output[global_idx] = diff >  127?  127: diff;\n\
    }\n\
    else{\n\
      output[global_idx] = 0;\n\
    }\n\
  }\n\
\n\
"
    kernel = kernel.replace("IMAGE_SIZE", self.image_height*self.image_width)
    kernel = kernel.replace("IMAGE_WIDTH", self.image_width)
    
    return kernel


  def process_frame(self, current_frame, previous_frame):
    self.current_frame  = cl_array.to_device(self.queue, current_frame.reshape(self.array_size))
    self.previous_frame = cl_array.to_device(self.queue, previous_frame.reshape(self.array_size))
    
    self.program(self.queue, self.global_work_shape, self.workgroup_shape,
                 self.current_frame_gpu.data, self.previous_frame_gpu.data,
                 self.result_gpu.data)
    self.queue.finish()
    
    result = self.result_gpu.get()
    
    pos = (result*(result > 0))
    neg = (numpy.abs(result)*(result < 0))
    
    return pos, neg



#######################################################################
def RoundUp(groupSize, globalSize):

  if  globalSize <= groupSize :
    return groupSize

  r = globalSize % groupSize
  if r == 0:
    return globalSize
  else:
    return globalSize + groupSize - r

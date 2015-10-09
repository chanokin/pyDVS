import pyopencl as cl
import pyopencl.array as cl_array
from pyopencl.algorithm import RadixSort
#from pyopencl.clrandom import RanluxGenerator
#import pyopencl.clrandom as cl_rand
import numpy
import sys
import os
os.environ['PYOPENCL_COMPILER_OUTPUT'] = '1'

    
class OpenCL_DVS():
  
  def __init__(self, image_width, image_height, threshold=0.01, 
               threshold_rate=0.01, max_threshold=64, min_threshold=8,
               starting_val=50, input_to_log=False, use_conv=False,
               cache_dir="./ocl_kernel_cache"):
    self.cache_dir = cache_dir
    self.image_width  = image_width
    self.image_height = image_height
    self.threshold = threshold
    self.threshold_rate = threshold_rate
    self.max_threshold = max_threshold
    self.min_threshold = min_threshold
    self.starting_val = starting_val
    self.input_to_log = input_to_log
    self.use_conv = use_conv
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
                                             cl_array.vec.short4)

    self.reference_frame_gpu = cl_array.zeros(self.queue, self.array_size,\
                                             cl_array.vec.short4)
    self.reference_frame_gpu.fill(cl_array.vec.make_short4(self.starting_val))
    
    #~ self.previous_frame_gpu = cl_rand.rand(self.queue, self.array_size,\
                                           #~ cl_array.vec.short4)
    #~ randGen = RanluxGenerator(self.queue)
    #~ randGen.fill_uniform(self.previous_frame_gpu, -32, 32)
    
    
        
    self.result_gpu  = cl_array.zeros(self.queue, self.array_size, \
                                      cl_array.vec.short4)
    
    self.program = self.init_program()
  

  def init_program(self):
    prog_src = self.kernel_str()
    program = cl.Program(self.context, prog_src)
    options = ['-Werror', 
               '-cl-mad-enable', 
               '-cl-single-precision-constant', 
               '-cl-no-signed-zeros', 
               '-cl-fast-relaxed-math',
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

  def centre_kern(self):
    centre = """
    float4 kc_0 = (float4)(0.06144015f, 0.46693352f, 0.91802812f, 0.46693352f),
           kc_1 = (float4)(0.06144015f, 0.91802812f, 0.46693352f, 0.06144015f);

           
    float4 curr01 = convert_float4(current[global_idx + 1]),
           curr10 = convert_float4(current[global_idx + IMAGE_WIDTH]),
           curr11 = convert_float4(current[global_idx + IMAGE_WIDTH + 1]),
           curr20 = convert_float4(current[global_idx + 2*IMAGE_WIDTH]),
           curr21 = convert_float4(current[global_idx + 2*IMAGE_WIDTH + 1]),
           curr30 = convert_float4(current[global_idx + 3*IMAGE_WIDTH]),
           curr31 = convert_float4(current[global_idx + 3*IMAGE_WIDTH + 1]),
           curr40 = convert_float4(current[global_idx + 4*IMAGE_WIDTH]),
           curr41 = convert_float4(current[global_idx + 4*IMAGE_WIDTH + 1]);
    
    float4 h_conv0, h_conv1, h_conv2, h_conv3, h_conv4;
    float4 tmp0, tmp1, tmp2, tmp3;

    tmp0 = kc_0*curr;
    tmp1 = kc_1*curr01;
    h_conv0.x = tmp0.x + tmp0.y + tmp0.z + tmp0.w + tmp1.x; // First of four col, first row 
    h_conv0.w = curr.w*kc_0.x + curr01.x*kc_0.y + tmp1.y + tmp1.z + tmp1.w; // Fourth of four col, first row 

    tmp0 = kc_0*curr10;
    tmp1 = kc_1*curr11;
    h_conv1.x = tmp0.x + tmp0.y + tmp0.z + tmp0.w + tmp1.x; // First of four col, second row 
    h_conv1.w = curr10.w*kc_0.x + curr11.x*kc_0.y + tmp1.y + tmp1.z + tmp1.w; // Fourth of four col, second row 

    tmp0 = kc_0*curr20;
    tmp1 = kc_1*curr21;
    h_conv2.x = tmp0.x + tmp0.y + tmp0.z + tmp0.w + tmp1.x; // First of four col, third row
    h_conv2.w = curr20.w*kc_0.x + curr21.x*kc_0.y + tmp1.y + tmp1.z + tmp1.w; // Fourth of four col, third row

    tmp0 = kc_0*curr30;
    tmp1 = kc_1*curr31;
    h_conv3.x = tmp0.x + tmp0.y + tmp0.z + tmp0.w + tmp1.x; // First of four col, fourth row
    h_conv3.w = curr30.w*kc_0.x + curr31.x*kc_0.y + tmp1.y + tmp1.z + tmp1.w; // Fourth of four col, fourth row 
    
    tmp0 = kc_0*curr40;
    tmp1 = kc_1*curr41;
    h_conv4.x = tmp0.x + tmp0.y + tmp0.z + tmp0.w + tmp1.x; // First of four col, fifth row
    h_conv4.w = curr40.w*kc_0.x + curr41.x*kc_0.y + tmp1.y + tmp1.z + tmp1.w; // Fourth of four col, fifth row 

///////////////////////////////////////////////////////////////////////////////////////////////////
    
    tmp2 = (float4)(curr.y, curr.z, curr.w, curr01.x);
    tmp3 = (float4)(curr01.y, curr01.y, curr01.z, curr01.w);
    tmp0 = kc_0*tmp2;
    tmp1 = kc_1*tmp3;
    h_conv0.y = tmp0.x + tmp0.y + tmp0.z + tmp0.w + tmp1.x; // Second of four col, first row
    h_conv0.z = tmp2.z*kc_0.x + tmp2.w*kc_0.y + tmp1.y + tmp1.z + tmp1.w; // Third of four col, first row 

    tmp2 = (float4)(curr10.y, curr10.z, curr10.w, curr11.x);
    tmp3 = (float4)(curr11.y, curr11.y, curr11.z, curr11.w);
    tmp0 = kc_0*tmp2;
    tmp1 = kc_1*tmp3;
    h_conv1.y = tmp0.x + tmp0.y + tmp0.z + tmp0.w + tmp1.x; // Second of four col, second row
    h_conv1.z = tmp2.w*kc_0.x + tmp2.w*kc_0.y + tmp1.y + tmp1.z + tmp1.w; // Third of four col, second row 

    tmp2 = (float4)(curr20.y, curr20.z, curr20.w, curr21.x);
    tmp3 = (float4)(curr21.y, curr21.y, curr21.z, curr21.w);
    tmp0 = kc_0*tmp2;
    tmp1 = kc_1*tmp3;
    h_conv2.y = tmp0.x + tmp0.y + tmp0.z + tmp0.w + tmp1.x; // Second of four col, second row
    h_conv2.z = tmp2.w*kc_0.x + tmp2.w*kc_0.y + tmp1.y + tmp1.z + tmp1.w; // Third of four col, second row 

    tmp2 = (float4)(curr30.y, curr30.z, curr30.w, curr31.x);
    tmp3 = (float4)(curr31.y, curr31.y, curr31.z, curr31.w);
    tmp0 = kc_0*tmp2;
    tmp1 = kc_1*tmp3;
    h_conv3.y = tmp0.x + tmp0.y + tmp0.z + tmp0.w + tmp1.x; // Second of four col, second row
    h_conv3.z = tmp2.w*kc_0.x + tmp2.w*kc_0.y + tmp1.y + tmp1.z + tmp1.w; // Third of four col, second row 

    tmp2 = (float4)(curr40.y, curr40.z, curr40.w, curr41.x);
    tmp3 = (float4)(curr41.y, curr41.y, curr41.z, curr41.w);
    tmp0 = kc_0*tmp2;
    tmp1 = kc_1*tmp3;
    h_conv4.y = tmp0.x + tmp0.y + tmp0.z + tmp0.w + tmp1.x; // Second of four col, second row
    h_conv4.z = tmp2.w*kc_0.x + tmp2.w*kc_0.y + tmp1.y + tmp1.z + tmp1.w; // Third of four col, second row 

    tmp2 = (float4)(h_conv0.x, h_conv1.x, h_conv2.x, h_conv3.x);
    tmp3 = (float4)(h_conv4.x, h_conv2.y, h_conv3.y, h_conv4.y);
    tmp0 = kc_0*tmp2;
    tmp1 = kc_1*tmp3;
    mod_curr.x = tmp0.x + tmp0.y + tmp0.z + tmp0.w + tmp1.x;
    mod_curr.y = h_conv0.y*kc_0.x + h_conv1.y*kc_0.y + tmp1.y + tmp1.z + tmp1.w;

    tmp2 = (float4)(h_conv0.z, h_conv1.z, h_conv2.z, h_conv3.z);
    tmp3 = (float4)(h_conv4.z, h_conv2.w, h_conv3.w, h_conv4.w);
    tmp0 = kc_0*tmp2;
    tmp1 = kc_1*tmp3;
    mod_curr.z = tmp0.x + tmp0.y + tmp0.z + tmp0.w + tmp1.x;
    mod_curr.w = h_conv0.w*kc_0.x + h_conv1.w*kc_0.y + tmp1.y + tmp1.z + tmp1.w;
    
"""

    return centre
  
  def surround_kern(self):
    surround = """
    kc_0 = (float4)(0.38311505f, 0.40082112f, 0.40690312f, 0.40082112f),
    kc_1 = (float4)(0.38311505f, 0.40690312f, 0.40082112f, 0.38311505f);
    tmp0 = kc_0*curr;
    tmp1 = kc_1*curr01;
    h_conv0.x = tmp0.x + tmp0.y + tmp0.z + tmp0.w + tmp1.x; // First of four col, first row 
    h_conv0.w = curr.w*kc_0.x + curr01.x*kc_0.y + tmp1.y + tmp1.z + tmp1.w; // Fourth of four col, first row 

    tmp0 = kc_0*curr10;
    tmp1 = kc_1*curr11;
    h_conv1.x = tmp0.x + tmp0.y + tmp0.z + tmp0.w + tmp1.x; // First of four col, second row 
    h_conv1.w = curr10.w*kc_0.x + curr11.x*kc_0.y + tmp1.y + tmp1.z + tmp1.w; // Fourth of four col, second row 

    tmp0 = kc_0*curr20;
    tmp1 = kc_1*curr21;
    h_conv2.x = tmp0.x + tmp0.y + tmp0.z + tmp0.w + tmp1.x; // First of four col, third row
    h_conv2.w = curr20.w*kc_0.x + curr21.x*kc_0.y + tmp1.y + tmp1.z + tmp1.w; // Fourth of four col, third row

    tmp0 = kc_0*curr30;
    tmp1 = kc_1*curr31;
    h_conv3.x = tmp0.x + tmp0.y + tmp0.z + tmp0.w + tmp1.x; // First of four col, fourth row
    h_conv3.w = curr30.w*kc_0.x + curr31.x*kc_0.y + tmp1.y + tmp1.z + tmp1.w; // Fourth of four col, fourth row 
    
    tmp0 = kc_0*curr40;
    tmp1 = kc_1*curr41;
    h_conv4.x = tmp0.x + tmp0.y + tmp0.z + tmp0.w + tmp1.x; // First of four col, fifth row
    h_conv4.w = curr40.w*kc_0.x + curr41.x*kc_0.y + tmp1.y + tmp1.z + tmp1.w; // Fourth of four col, fifth row 

///////////////////////////////////////////////////////////////////////////////////////////////////
    
    tmp2 = (float4)(curr.y, curr.z, curr.w, curr01.x);
    tmp3 = (float4)(curr01.y, curr01.y, curr01.z, curr01.w);
    tmp0 = kc_0*tmp2;
    tmp1 = kc_1*tmp3;
    h_conv0.y = tmp0.x + tmp0.y + tmp0.z + tmp0.w + tmp1.x; // Second of four col, first row
    h_conv0.z = tmp2.z*kc_0.x + tmp2.w*kc_0.y + tmp1.y + tmp1.z + tmp1.w; // Third of four col, first row 

    tmp2 = (float4)(curr10.y, curr10.z, curr10.w, curr11.x);
    tmp3 = (float4)(curr11.y, curr11.y, curr11.z, curr11.w);
    tmp0 = kc_0*tmp2;
    tmp1 = kc_1*tmp3;
    h_conv1.y = tmp0.x + tmp0.y + tmp0.z + tmp0.w + tmp1.x; // Second of four col, second row
    h_conv1.z = tmp2.w*kc_0.x + tmp2.w*kc_0.y + tmp1.y + tmp1.z + tmp1.w; // Third of four col, second row 

    tmp2 = (float4)(curr20.y, curr20.z, curr20.w, curr21.x);
    tmp3 = (float4)(curr21.y, curr21.y, curr21.z, curr21.w);
    tmp0 = kc_0*tmp2;
    tmp1 = kc_1*tmp3;
    h_conv2.y = tmp0.x + tmp0.y + tmp0.z + tmp0.w + tmp1.x; // Second of four col, second row
    h_conv2.z = tmp2.w*kc_0.x + tmp2.w*kc_0.y + tmp1.y + tmp1.z + tmp1.w; // Third of four col, second row 

    tmp2 = (float4)(curr30.y, curr30.z, curr30.w, curr31.x);
    tmp3 = (float4)(curr31.y, curr31.y, curr31.z, curr31.w);
    tmp0 = kc_0*tmp2;
    tmp1 = kc_1*tmp3;
    h_conv3.y = tmp0.x + tmp0.y + tmp0.z + tmp0.w + tmp1.x; // Second of four col, second row
    h_conv3.z = tmp2.w*kc_0.x + tmp2.w*kc_0.y + tmp1.y + tmp1.z + tmp1.w; // Third of four col, second row 

    tmp2 = (float4)(curr40.y, curr40.z, curr40.w, curr41.x);
    tmp3 = (float4)(curr41.y, curr41.y, curr41.z, curr41.w);
    tmp0 = kc_0*tmp2;
    tmp1 = kc_1*tmp3;
    h_conv4.y = tmp0.x + tmp0.y + tmp0.z + tmp0.w + tmp1.x; // Second of four col, second row
    h_conv4.z = tmp2.w*kc_0.x + tmp2.w*kc_0.y + tmp1.y + tmp1.z + tmp1.w; // Third of four col, second row 

    tmp2 = (float4)(h_conv0.x, h_conv1.x, h_conv2.x, h_conv3.x);
    tmp3 = (float4)(h_conv4.x, h_conv2.y, h_conv3.y, h_conv4.y);
    tmp0 = kc_0*tmp2;
    tmp1 = kc_1*tmp3;
    mod_curr.x -= tmp0.x + tmp0.y + tmp0.z + tmp0.w + tmp1.x;
    mod_curr.y -= h_conv0.y*kc_0.x + h_conv1.y*kc_0.y + tmp1.y + tmp1.z + tmp1.w;

    tmp2 = (float4)(h_conv0.z, h_conv1.z, h_conv2.z, h_conv3.z);
    tmp3 = (float4)(h_conv4.z, h_conv2.w, h_conv3.w, h_conv4.w);
    tmp0 = kc_0*tmp2;
    tmp1 = kc_1*tmp3;
    mod_curr.z -= tmp0.x + tmp0.y + tmp0.z + tmp0.w + tmp1.x;
    mod_curr.w -= h_conv0.w*kc_0.x + h_conv1.w*kc_0.y + tmp1.y + tmp1.z + tmp1.w;
    
    curr = mod_curr;
"""
    return surround 

  def processing_str(self):
    if self.use_conv:
      proc = "%s\n%s"%(self.centre_kern(), self.surround_kern())
    else:
      if self.input_to_log:
        proc = """
    mod_curr = log(curr + 1.0f);
    mod_ref  = log(ref + 1.0f);
"""
      else:
        proc = """
    mod_curr = curr;
    mod_ref  = ref;
"""
    proc = "%s\n%s\n"%(proc, "    diff = mod_curr - mod_ref;")
      
    return proc

  def kernel_str(self):      
    kernel = """
#pragma OPENCL EXTENSION cl_amd_media_ops : enable

__kernel void difference(__global short4* current, __global short4* reference,
                         __global short4*  output){
  uint row = get_global_id(1), col = get_global_id(0);

  if(row < IMAGE_HEIGHT && col < IMAGE_WIDTH){
    uint global_idx = row*IMAGE_WIDTH + col;
    float4 gt_thresh = 0.0f, one = 1.0f, zero = 0.0f;
    float4 curr, ref, diff, mod_curr, mod_ref;
  
    curr = convert_float4(current[global_idx]);
    ref  = convert_float4(reference[global_idx]);
    
    DIFF_OP
    
    gt_thresh = diff*diff > SQUARED_THRESHOLD ? one : gt_thresh;
    
    diff = gt_thresh > zero ? diff : zero;
    ref  = gt_thresh > zero ? curr : ref;

    output[global_idx]    = convert_short4(diff);
    reference[global_idx] = convert_short4(ref);
  }
}
"""

    if self.input_to_log:
      scaled_threshold = 0.0#(numpy.log(255)*self.threshold)**2
    else:
      scaled_threshold = (255*self.threshold)**2
    

    kernel = kernel.replace("DIFF_OP", self.processing_str())
    kernel = kernel.replace("IMAGE_SIZE", str(self.image_height*self.image_width))
    kernel = kernel.replace("IMAGE_HEIGHT", str(self.image_height))
    kernel = kernel.replace("IMAGE_WIDTH", str(self.image_width))
    kernel = kernel.replace("SQUARED_THRESHOLD", str(scaled_threshold))
    kernel = kernel.replace("MIN_THRESHOLD", str(self.min_threshold))

    print kernel
    return kernel
    
    
  def process_frame(self, current_frame):
    '''Returns difference value and sorted indices for further processing'''
    
    self.current_frame_gpu.set(current_frame.astype(cl_array.vec.short4))

    self.program.difference(self.queue, self.global_work_shape, self.workgroup_shape,
                            self.current_frame_gpu.data, self.reference_frame_gpu.data,
                            self.result_gpu.data)
    self.queue.finish()
    diff = self.result_gpu.get().astype(numpy.int16)

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

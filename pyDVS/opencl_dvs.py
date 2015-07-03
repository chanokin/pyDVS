import pyopencl as cl
import pyopencl.array as cl_array
from pyopencl.algorithm import RadixSort
import numpy
import sys
import os
os.environ['PYOPENCL_COMPILER_OUTPUT'] = '1'

    
class OpenCL_DVS():
  
  def __init__(self, image_width, image_height, threshold=0.01, 
               threshold_rate=0.01, max_threshold=64, min_threshold=8,
               cache_dir="./ocl_kernel_cache"):
    self.cache_dir = cache_dir
    self.image_width  = image_width
    self.image_height = image_height
    self.threshold = threshold
    self.threshold_rate = threshold_rate
    self.max_threshold = max_threshold
    self.min_threshold = min_threshold
    
    self.workgroup_shape = (8, 8)
    self.global_work_shape = (RoundUp(self.workgroup_shape[0], self.image_width),
                              RoundUp(self.workgroup_shape[1], self.image_height))

    self.array_size = self.image_width*self.image_height
    
    self.context, self.device = self.create_context()

    if self.context == None:
      print "Failed to create OpenCL context."
      return 1
    
    self.queue = cl.CommandQueue(self.context, self.device)

    self.threshold_gpu  = cl_array.zeros(self.queue, self.array_size,\
                                         cl_array.vec.uchar4)
    self.threshold_gpu.fill(cl_array.vec.make_uchar4(threshold))
    
    self.current_frame_gpu  = cl_array.zeros(self.queue, self.array_size,\
                                             cl_array.vec.uchar4)
    self.previous_frame_gpu = cl_array.zeros(self.queue, self.array_size,\
                                             cl_array.vec.uchar4)
    self.previous_frame_gpu.fill(cl_array.vec.make_uchar4(0))
    
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


  def kernel_str(self):      
    kernel = """
#pragma OPENCL EXTENSION cl_amd_media_ops : enable

__kernel void difference(__global uchar4* current, __global uchar4* previous,
                         __global uchar4* threshold, __global short4*  output){
  uint row = get_global_id(1), col = get_global_id(0);

  //from basab's thesis off-centre
//*
  float k00 = -0.27399398, k01 = 0.06762399, k02 = -0.27399398,
        k10 =  0.06762399, k11 = 0.82547997, k12 =  0.06762399,
        k20 = -0.27399398, k21 = 0.06762399, k22 = -0.27399398;
//*/

/*
  float k00 = -1./12., k01 = -1./12., k02 = -1./12.,
        k10 = -1./12., k11 =  1./3.,  k12 = -1./12.,
        k20 = -1./12., k21 = -1./12., k22 = -1./12.;
*/
/*
  float k00 =  0,      k01 = -1./12., k02 =  0,
        k10 = -1./12., k11 =  2./3.,  k12 = -1./12.,
        k20 =  0,      k21 = -1./12., k22 =  0;
*/
/*
  float k00 = -1./8., k01 = -1./8., k02 = -1./8.,
        k10 = -1./8., k11 = 1.,     k12 = -1./8.,
        k20 = -1./8., k21 = -1./8., k22 = -1./8.;
*/

  if(row > 1 && col > 1 && row < IMAGE_HEIGHT_m_2 && col < IMAGE_WIDTH_m_2){
    uint global_idx = row*IMAGE_WIDTH + col;
    int4 gt_thresh = 0, true_vec = 1;
    int4 curr, prev, diff, thre;
  
    int4 curr_conv_row0[2],
         curr_conv_row1[2],
         curr_conv_row2[2];
    
    int4 conv;
    
    float sum = 0;
    
    curr_conv_row0[0] = convert_int4(current[global_idx - IMAGE_WIDTH]);
    curr_conv_row0[1] = convert_int4(current[global_idx - IMAGE_WIDTH_p_1]);
    curr_conv_row1[0] = convert_int4(current[global_idx]);
    curr_conv_row1[1] = convert_int4(current[global_idx + 1]);
    curr_conv_row2[0] = convert_int4(current[global_idx + IMAGE_WIDTH]);
    curr_conv_row2[1] = convert_int4(current[global_idx + IMAGE_WIDTH_p_1]);

/*
    curr_conv_row0[0] -= convert_int4(previous[global_idx - IMAGE_WIDTH]);
    curr_conv_row0[1] -= convert_int4(previous[global_idx - IMAGE_WIDTH_p_1]);
    curr_conv_row1[0] -= convert_int4(previous[global_idx]);
    curr_conv_row1[1] -= convert_int4(previous[global_idx + 1]);
    curr_conv_row2[0] -= convert_int4(previous[global_idx + IMAGE_WIDTH]);
    curr_conv_row2[1] -= convert_int4(previous[global_idx + IMAGE_WIDTH_p_1]);
*/
/*
    curr_conv_row0[0] = convert_int4(abs(curr_conv_row0[0]));
    curr_conv_row0[1] = convert_int4(abs(curr_conv_row0[1]));
    curr_conv_row1[0] = convert_int4(abs(curr_conv_row1[0]));
    curr_conv_row1[1] = convert_int4(abs(curr_conv_row1[1]));
    curr_conv_row2[0] = convert_int4(abs(curr_conv_row2[0]));
    curr_conv_row2[1] = convert_int4(abs(curr_conv_row2[1]));
*/ 

    curr = convert_int4(current[global_idx]);
    prev = convert_int4(previous[global_idx]);
    thre = convert_int4(threshold[global_idx]);
    
    sum = 0;
    sum =  curr_conv_row0[0].x*k00 + curr_conv_row0[0].y*k01 + curr_conv_row0[0].z*02;
    //sum += curr_conv_row1[0].x*k10 + curr.x*k11              + curr_conv_row1[0].z*k12;
    sum += curr_conv_row1[0].x*k10 + curr_conv_row1[0].y*k11 + curr_conv_row1[0].z*k12;
    sum += curr_conv_row2[0].x*k20 + curr_conv_row2[0].y*k21 + curr_conv_row2[0].z*k22;
    conv.x = sum;
    
    
    sum = 0;
    sum =  curr_conv_row0[0].y*k00 + curr_conv_row0[0].z*k01 + curr_conv_row0[0].w*k02;
    //sum += curr_conv_row1[0].y*k10 + curr.y*k11              + curr_conv_row1[0].w*k12;
    sum += curr_conv_row1[0].y*k10 + curr_conv_row1[0].z*k11 + curr_conv_row1[0].w*k12;
    sum += curr_conv_row2[0].y*k20 + curr_conv_row2[0].z*k21 + curr_conv_row2[0].w*k22;
    conv.y = sum;
    
    
    sum = 0;
    sum =  curr_conv_row0[0].z*k00 + curr_conv_row0[0].w*k01 + curr_conv_row0[1].x*k02;
    //sum += curr_conv_row1[0].z*k10 + curr.z*k11              + curr_conv_row1[1].x*k12;
    sum += curr_conv_row1[0].z*k10 + curr_conv_row1[0].w*k11 + curr_conv_row1[1].x*k12;
    sum += curr_conv_row2[0].z*k20 + curr_conv_row2[0].w*k21 + curr_conv_row2[1].x*k22;
    conv.z = sum;
    
    
    sum = 0;
    sum =  curr_conv_row0[0].w*k00 + curr_conv_row0[1].x*k01 + curr_conv_row0[1].y*k02;
    //sum += curr_conv_row1[0].w*k10 + curr.w*k11              + curr_conv_row1[1].y*k12;
    sum += curr_conv_row1[0].w*k10 + curr_conv_row1[1].x*k11 + curr_conv_row1[1].y*k12;
    sum += curr_conv_row2[0].w*k20 + curr_conv_row2[1].x*k21 + curr_conv_row2[1].y*k22;
    conv.w = sum;
    
    
    curr = conv;

    diff = curr - prev;
    //diff = conv - prev;
    //curr = conv;
    //diff = conv;
    
    gt_thresh = diff*diff > thre*thre ? true_vec : gt_thresh;
    
    diff = gt_thresh > 0 ? diff : 0;
    
    thre = gt_thresh > 0 ? thre + THRESHOLD_RATE_UP : thre - THRESHOLD_RATE_DOWN;
    thre = thre > MAX_THRESHOLD ? MAX_THRESHOLD : thre;
    thre = thre < MIN_THRESHOLD ? MIN_THRESHOLD : thre;

    output[global_idx]    = convert_short4(diff);
    threshold[global_idx] = convert_uchar4(thre);
    previous[global_idx]  = convert_uchar4(curr);
  }
}
"""
    #~ kernel = """
#~ #pragma OPENCL EXTENSION cl_amd_media_ops : enable
#~ 
#~ __kernel void difference(__global uchar4* current, __global uchar4* previous,
                         #~ __global uchar4* threshold, __global short4*  output){
  #~ uint row = get_global_id(1), col = get_global_id(0);
#~ 
#~ 
  #~ if(row < IMAGE_HEIGHT_m_2 && col < IMAGE_WIDTH_m_2){
    #~ uint global_idx = row*IMAGE_WIDTH + col;
    #~ int4 gt_thresh = 0, true_vec = 1;
    #~ int4 curr, prev, diff, thre;
  #~ 
    #~ curr = convert_int4(current[global_idx]);
    #~ prev = convert_int4(previous[global_idx]);
    #~ thre = convert_int4(threshold[global_idx]);
    #~ diff = curr - prev;
    #~ 
    #~ gt_thresh = diff*diff > thre*thre ? true_vec : gt_thresh;
    #~ 
    #~ diff = gt_thresh > 0 ? diff : 0;
    #~ 
    #~ thre = gt_thresh > 0 ? thre + THRESHOLD_RATE_UP : thre - THRESHOLD_RATE_DOWN;
    #~ thre = thre > MAX_THRESHOLD ? MAX_THRESHOLD : thre;
    #~ thre = thre < MIN_THRESHOLD ? MIN_THRESHOLD : thre;
#~ 
    #~ output[global_idx]    = convert_short4(diff);
    #~ threshold[global_idx] = convert_uchar4(thre);
    #~ previous[global_idx]  = convert_uchar4(curr);
  #~ }
#~ }
#~ """
    kernel = kernel.replace("IMAGE_SIZE", str(self.image_height*self.image_width))
    kernel = kernel.replace("IMAGE_WIDTH_p_1", str(self.image_width + 1))
    kernel = kernel.replace("IMAGE_WIDTH_t_2_p_1", str(self.image_width*2 + 1))
    kernel = kernel.replace("IMAGE_HEIGHT_m_2", str(self.image_height - 2))
    kernel = kernel.replace("IMAGE_WIDTH_m_2", str(self.image_width - 2))
    kernel = kernel.replace("IMAGE_HEIGHT", str(self.image_height))
    kernel = kernel.replace("IMAGE_WIDTH", str(self.image_width))
    kernel = kernel.replace("THRESHOLD_RATE_UP", str(self.threshold_rate))
    kernel = kernel.replace("THRESHOLD_RATE_DOWN", str(self.threshold_rate))
    kernel = kernel.replace("MAX_THRESHOLD", str(self.max_threshold))
    kernel = kernel.replace("MIN_THRESHOLD", str(self.min_threshold))

    #print kernel
    return kernel
    
    
  def process_frame(self, current_frame):
    '''Returns difference value and sorted indices for further processing'''
    
    self.current_frame_gpu.set(current_frame.astype(cl_array.vec.uchar4))

    self.program.difference(self.queue, self.global_work_shape, self.workgroup_shape,
                            self.current_frame_gpu.data, self.previous_frame_gpu.data,
                            self.threshold_gpu.data, self.result_gpu.data)
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

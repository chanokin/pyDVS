
#pragma OPENCL EXTENSION cl_amd_media_ops : enable
#pragma OPENCL EXTENSION cl_khr_byte_addressable_store : enable

__kernel void difference(__global uchar* current, __global uchar* previous,
                         __global short*  output){
  uint col = get_global_id(0), row = get_global_id(1),
       global_idx = row*640 + col;
       
  int curr, prev, diff;

  if(global_idx < 230400){
    curr = current[global_idx];
    prev = previous[global_idx];
    diff = curr - prev;
    if( diff*diff > 5776 ){
      output[global_idx] = diff;
    }
    else{
      output[global_idx] = -10;
    }
  }
}


__constant int pyopencl_defeat_cache_8ec5df03f43b423d963807f163c0913f = 0;
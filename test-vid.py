#!/usr/bin/python

import numpy
import cv
import cv2
import sys
import time
from os.path import basename, splitext
from pyDVS import pyDVS


blue, green, red = range(3)
threshold = 0.0
starting_val = 0
threshold_rate = 0.0
max_threshold = 96
min_threshold = 8
max_count = 300
#~ max_count = 9000000000000
max_dummy_count = 0
dummy_count = 0 
count = 0
frame_available = True
grayscale_video = False
input_to_log = False
use_conv = True

#force_numpy=True
force_numpy=False
#video_device = sys.argv[1]
#~ video_device = "../../sources/paprika.mp4"
#video_device = "../../sources/visor_1203702621921_competition_short.avi"
#~ video_device = "../../sources/visor_1203702802078_Camera2_070605.avi"
#~ video_device = "../../sources/bowing_cif.y4m"
#~ video_device = "../../sources/120fpsHFRSample.mp4"
#~ video_device = "../../sources/GoPro HD Hero 2 120fps Slow Motion.mp4"
video_device = "../../sources/iPhone 5s - 120fps Slow Motion Ultimate Video Test.mp4"
#~ video_device = "../../sources/action_youtube_naudio/walking/v_walk_dog_12/v_walk_dog_12_04.avi"
mydvs = pyDVS(video_device, force_numpy=force_numpy, threshold=threshold,
              threshold_rate=threshold_rate, max_threshold=max_threshold,
              min_threshold=min_threshold, starting_val=starting_val, 
              input_to_log=input_to_log, use_conv=use_conv)
width = mydvs.frame_width
height = mydvs.frame_height

fps = mydvs.frames_per_second
fourcc = cv.CV_FOURCC(*'XVID')
#output_file = sys.argv[2]
video_name = "%s"%(splitext(basename(video_device))[0])
log_or_linear = "log" if input_to_log else "linear"
output_file = "output_thres-%s_%s_%s"%(threshold, video_name, log_or_linear)
video_writer = cv2.VideoWriter('%s.avi'%(output_file),fourcc, fps, 
                               (width, height), not grayscale_video)
composite = numpy.zeros((height, width, 3), dtype=numpy.uint8)
print(composite.shape)

encoding_times = []
while frame_available and count < max_count:
  prev_time = time.time()
  frame_available = mydvs.update()
  end_time = time.time()
  
  encoding_times.append(end_time - prev_time)
  
  if dummy_count < max_dummy_count:
    dummy_count += 1
  else:
    composite[:,:,:] = 0
    curr_frame = mydvs.current_frame
    res_frame = mydvs.result_frame
    curr_frame[numpy.where(res_frame > 0)] = 0
    curr_frame[numpy.where(res_frame < 0)] = 0
    
    pos_frame = 255*(res_frame>0).astype(numpy.uint8)
    neg_frame = 255*(res_frame<0).astype(numpy.uint8)
    
    composite[:,:, red] = (curr_frame + neg_frame).reshape((height, width)) 
    composite[:,:, green] = (curr_frame + pos_frame).reshape((height, width))
    composite[:,:, blue] = curr_frame.reshape((height, width))
    
    #~ composite[:,:, red] = (neg_frame).reshape((height, width)) 
    #~ composite[:,:, green] = ( pos_frame).reshape((height, width))
    
    
    #~ composite[:,:, blue] = mydvs.processor.threshold.reshape((height, width))
    
    #~ composite[:,:, blue] = 255*(res_frame > 0).reshape((height, width))
    #~ composite[:,:, red] = 255*(res_frame < 0).reshape((height, width))
    #~ composite[:,:, red] = res_frame.reshape((height, width))
    
    video_writer.write(composite)
    
    count += 1 
    if count%100==0:
      print("Progress!!!")
      #~ print(numpy.sum(res_frame > 0))
      #~ print(numpy.sum(res_frame < 0))

  

average_encoding_time = numpy.average(encoding_times)
video_writer.release()
cv2.destroyAllWindows()


print("Average encoding time = %s"%average_encoding_time)

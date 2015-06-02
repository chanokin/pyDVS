#!/usr/bin/python

import numpy
import cv
import cv2
import sys
import time

from pyDVS import pyDVS


blue, green, red = range(3)

mydvs = pyDVS(sys.argv[1], force_numpy=True, threshold=0.3)
width = mydvs.frame_width
height = mydvs.frame_height

grayscale_video = False
fps = mydvs.frames_per_second
fourcc = cv.CV_FOURCC(*'XVID')
video_writer = cv2.VideoWriter('%s.avi'%(sys.argv[2]),fourcc, fps, 
                               (width, height), not grayscale_video)
composite = numpy.zeros((height, width, 3), dtype=numpy.uint8)

max_count = 1000#000000
max_dummy_count = 0
dummy_count = 0 
count = 0
frame_available = True
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

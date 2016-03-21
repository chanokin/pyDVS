from __future__ import print_function
import os
import glob
import threading
import time

import numpy
from numpy import int16, uint16, uint8, float16, log2
DTYPE = int16

import cv2
from cv2 import cvtColor as convertColor, COLOR_BGR2GRAY, COLOR_GRAY2RGB,\
                resize

try:                  #nearest neighboor interpolation
  from cv2.cv import CV_INTER_NN, \
                     CV_CAP_PROP_FRAME_WIDTH, \
                     CV_CAP_PROP_FRAME_HEIGHT, \
                     CV_CAP_PROP_FPS
except:
  from cv2 import INTER_NEAREST as CV_INTER_NN, \
                  CAP_PROP_FRAME_WIDTH as CV_CAP_PROP_FRAME_WIDTH, \
                  CAP_PROP_FRAME_HEIGHT as CV_CAP_PROP_FRAME_HEIGHT, \
                  CAP_PROP_FPS as CV_CAP_PROP_FPS

from pydvs.virtual_cam import VirtualCam
#BEHAVE_MICROSACCADE = "SACCADE"
#BEHAVE_ATTENTION    = "ATTENTION"
#BEHAVE_TRAVERSE     = "TRAVERSE"
#BEHAVE_FADE         = "FADE"

fps = 90

max_frame_time = 1./fps
resolution=128
behaviour = VirtualCam.BEHAVE_TRAVERSE
vcam = VirtualCam("./mnist/", fps=fps, resolution=resolution, behaviour=behaviour)
valid_img = True
img = numpy.zeros((resolution, resolution), dtype=uint8)
ref = numpy.zeros((resolution, resolution), dtype=int16)
print(vcam.image_filenames)
frame_count = 0
start_time = time.time()
prev_time = time.time()
wait_time = 0.
while True:
  
  valid_img, img[:] = vcam.read(ref)
  #ref[:] = img

  cv2.imshow("label", img)
  
  key = cv2.waitKey(1) & 0xFF
  
  if key == ord('q') or key == ord('Q'):
    break

  frame_count += 1
  now = time.time()
  wait_time = max_frame_time - (now - prev_time)
  if wait_time > 0:
    time.sleep(wait_time)
  
  if now - start_time >= 1.:
    print("FPS: %d"%frame_count)
    frame_count = 0
    start_time = now
  
  prev_time = time.time()
  

vcam.stop()

from __future__ import print_function
import time

import numpy as np
from numpy import int16, uint8
DTYPE = int16

import cv2

from pydvs.virtual_cam import VirtualCam
#BEHAVE_MICROSACCADE = "SACCADE"
#BEHAVE_ATTENTION    = "ATTENTION"
#BEHAVE_TRAVERSE     = "TRAVERSE"
#BEHAVE_FADE         = "FADE"

fps = 120
max_cycles = 1

max_frame_time = 1./fps
resolution=32
behaviour = VirtualCam.BEHAVE_MICROSACCADE
on_ms = 1000.
off_ms = on_ms*2.
vcam = VirtualCam("./mnist/t10k/", fps=fps, resolution=resolution, behaviour=behaviour,
                  image_on_time_ms = on_ms, inter_off_time_ms = off_ms, max_cycles=max_cycles)
valid_img = True
img = np.zeros((resolution, resolution), dtype=uint8)
ref = np.zeros((resolution, resolution), dtype=int16)
# print(vcam.image_filenames)
frame_count = 0
start_time = time.time()
prev_time = time.time()
wait_time = 0.
WINDOW_NAME = 'MNIST Images'
cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_AUTOSIZE)
cv2.startWindowThread()

while True:

  valid_img, img[:] = vcam.read(ref)
  #ref[:] = img
  
  if not valid_img:
    if max_cycles == 1:
      print("Finished the specified single cycle")
    else:
      print("Finished the specified %i cycles" %max_cycles)
    break

  cv2.imshow(WINDOW_NAME, img)

  key = cv2.waitKey(10) & 0xFF
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
cv2.destroyAllWindows()
cv2.waitKey(1)

import time

import numpy as np
from numpy import int16, uint8, log2

import cv2

from pydvs.nvs import NVSEmu
from pydvs.visualization import render_frame
from pydvs.pdefines import *

import argparse


parser = argparse.ArgumentParser()
parser.add_argument("--video_id", default='0', required=False, type=str)
# parser.add_argument("--res", default=MODE_128, required=False, type=int)
args = parser.parse_args()
video_dev_id = args.video_id

if len(video_dev_id) < 4:
    # Assume that urls have at least 4 characters
    video_dev_id = int(video_dev_id)

video_dev = cv2.VideoCapture(video_dev_id)  # webcam
# video_dev = cv2.VideoCapture('/path/to/video/file') # movie

print("Is video device open? %s" % video_dev.isOpened())


width = 640
height = 480
video_dev.set(cv2.CAP_PROP_FRAME_WIDTH, width)
video_dev.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
# width = DTYPE_IDX( video_dev.get(cv2.CAP_PROP_FRAME_WIDTH) )
# height = DTYPE_IDX( video_dev.get(cv2.CAP_PROP_FRAME_HEIGHT) )
shape = (height, width)

# 640x480 seems to be locked-ish at 30fps (with good light)
# fps = 30
# low light gets the fps to 15
fps = 15

# fps = video_dev.get(cv2.CAP_PROP_FPS)

raw = np.zeros((height, width, 3), dtype='uint8')
curr = np.zeros(shape, dtype=DTYPE)
spikes_on = curr.copy()
spikes_off = curr.copy()

max_time_ms = int(1000./float(fps))



nvs_config = {
    'reference': {
        'dec': DTYPE( np.exp(-1.0 / 30.0) ), # frames
    },
    'threshold':{
        'dec': DTYPE( np.exp(-1.0 / 30.0) ), # frames
        'increment': DTYPE( 1.5 ), # mult
        'base': DTYPE( 0.05 * 255.0 ), # v
    },
    'range':{
        'minimum': DTYPE( 0.0 ), # v
        'maximum': DTYPE( 255.0 ), # v
    },
}
nvs = NVSEmu(shape, scale=False, config=nvs_config)

#---------------------- main loop -------------------------------------#

WINDOW_NAME = 'spikes'
cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_AUTOSIZE)
cv2.startWindowThread()

is_first_pass = True
start_time = time.time()
end_time = 0
frame_count = 0
while(True):
    # get an image from video source
    check, raw[:] = video_dev.read()

    if not check:
        print("A problem with frame acquisition occurred. Exiting!")
        break

    curr[:] = DTYPE( cv2.cvtColor(raw, cv2.COLOR_BGR2GRAY) )
    
    nvs.update(curr)
    spikes_on[:], spikes_off[:] = nvs.spikes()

    raw[:] = render_frame(spikes_on - spikes_off, curr, width, height)
    
    cv2.imshow(WINDOW_NAME, raw)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    end_time = time.time()

    if end_time - start_time >= 1.0:
        print("%d frames per second" % (frame_count))
        frame_count = 0
        start_time = time.time()
    else:
        frame_count += 1

cv2.destroyAllWindows()
cv2.waitKey(1)

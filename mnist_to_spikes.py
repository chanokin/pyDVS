from __future__ import print_function
import numpy as np
from numpy import int16, uint8
import cv2
import pylab
from datetime import datetime
from time import time, sleep
from pydvs.virtual_cam import *

import pyximport; pyximport.install()
from pydvs.generate_spikes import *

KEY_SPINNAKER = 0

MODE_128 = "128"
MODE_64  = "64"
MODE_32  = "32"
MODE_16  = "16"

UP_POLARITY     = "UP"
DOWN_POLARITY   = "DOWN"
MERGED_POLARITY = "MERGED"
RECTIFIED_POLARITY = "RECTIFIED"
POLARITY_DICT   = {UP_POLARITY: uint8(0), 
                 DOWN_POLARITY: uint8(1), 
                 MERGED_POLARITY: uint8(2),
                 RECTIFIED_POLARITY: uint8(3),
                 0: UP_POLARITY,
                 1: DOWN_POLARITY,
                 2: MERGED_POLARITY,
                 3: RECTIFIED_POLARITY}

OUTPUT_RATE         = "RATE"
OUTPUT_TIME         = "TIME"
OUTPUT_TIME_BIN     = "TIME_BIN"
OUTPUT_TIME_BIN_THR = "TIME_BIN_THR"

BEHAVE_MICROSACCADE = "SACCADE"
BEHAVE_ATTENTION    = "ATTENTION"
BEHAVE_TRAVERSE     = "TRAVERSE"
BEHAVE_FADE         = "FADE"

IMAGE_TYPES = ["png", 'jpeg', 'jpg']


def wait_ms(prev_time_ms, wait_time_ms):
    t = time.time()*1000. - prev_time_ms
    while t < wait_time_ms:
        if t < 0:
            break
        sleep((wait_time_ms - t)/1000.)
        t = time.time()*1000. - prev_time_ms
        
def get_filenames(img_dir):
    print("grabbing images from '%s'"%(img_dir))
    imgs = []
    img_idx = 0
    image_list = glob.glob(os.path.join(img_dir, "*.png"))

    image_list.sort()
    for img in image_list:
        if os.path.isfile(img):
            imgs.append(img)

    return imgs

setname = "t10k"
# setname = "train"
orig_w = 28
cam_w = 32
cam_fps = 30
frame_time_ms = np.round(1000./cam_fps)
frames_per_image = 90
on_time_ms = frame_time_ms*frames_per_image
off_time_ms = on_time_ms*3
frames_off = int(off_time_ms/cam_fps)
img_idx = 1
start_img_idx = 0
num_images = 60000 if setname == 'train' else 10000
num_cycles = 1
frames_per_saccade = cam_fps/3 - 1
frames_per_microsaccade = 2
polarity = POLARITY_DICT[RECTIFIED_POLARITY]
polarity = POLARITY_DICT[MERGED_POLARITY]
output_type = OUTPUT_TIME
history_weight = 1.
behaviour = VirtualCam.BEHAVE_MICROSACCADE

data_shift = uint8(np.log2(cam_w))
flag_shift = uint8(2*data_shift)
data_mask  = uint8(cam_w - 1)

inh_width = 2
is_inh_on = False
inh_coords = generate_inh_coords(cam_w, cam_w, inh_width)


spk_fname = "mnist_spikes/%s/mnist_%s_inh_%s___%dfps_%dx%d_res__img_%%05d.txt"%\
            (setname, setname, is_inh_on, cam_fps, cam_w, cam_w)

orig_img = np.zeros((orig_w, orig_w), dtype=int16)
padd_img = np.zeros((cam_w, cam_w), dtype=int16)
ref = np.zeros((cam_w, cam_w), dtype=int16)
curr = np.zeros((cam_w, cam_w), dtype=int16)
diff = np.zeros((cam_w, cam_w), dtype=int16)
abs_diff = np.zeros((cam_w, cam_w), dtype=int16)
spikes = np.zeros((cam_w, cam_w), dtype=int16)
spk_img = np.zeros((cam_w, cam_w, 3), dtype=uint8)
thresh = int( (2**8 - 1)*0.06 )
frm = (cam_w - orig_w)/2.
to  = frm + orig_w

write2file_count = 0
num_imgs_to_write = 100
base_time = 0
t = 0
valid = False
neg = None
pos = None
max_diff = 0
lists = []

write_buff = []
prev_ms = time.time()*1000.
start_ms = time.time()*1000.
image_paths = get_filenames('./mnist/%s'%(setname))
cx = 0
cy = 0
max_dist = 1
frames_per_microsaccade = 2
bg_gray = 0
filename = ""
fade_mask = imread("fading_mask.png", CV_LOAD_IMAGE_GRAYSCALE)
fade_mask = numpy.float64(fade_mask/(255.))
num_images = len(image_paths)
print("number of images found %s"%(num_images))
for img_idx in range(num_images):
    filename = image_paths[img_idx]
    orig_img[:] = imread(filename, CV_LOAD_IMAGE_GRAYSCALE)
    padd_img[:] = 0
    padd_img[frm:to, frm:to] = orig_img 
    # padd_img *= fade_mask
    cx = 0
    cy = 0
    t = 0
    for img_on_frame in range(frames_per_image):
        
        if img_on_frame == 0:
            curr[:] = padd_img
        else:
            curr[:], cx, cy = usaccade_image(padd_img, img_on_frame, 
            # curr[:], cx, cy = usaccade_image(curr, img_on_frame, 
                                             frames_per_microsaccade,
                                             max_dist, cx, cy, bg_gray)
        
        curr = int16(curr*fade_mask)
        
        diff[:], abs_diff[:], spikes[:] = thresholded_difference(curr, ref, thresh)
        
        if is_inh_on:
            spikes[:] = local_inhibition(spikes, abs_diff, inh_coords, 
                                         cam_w, cam_w, inh_width)

        ref[:] = update_reference_time_thresh(abs_diff, spikes, ref,
                                              thresh,
                                              frame_time_ms,
                                              history_weight)
        
        neg, pos, max_diff = split_spikes(spikes, abs_diff, polarity)

        lists[:] = []
        lists[:] = make_spike_lists_time(pos, neg, max_diff,
                                         flag_shift, data_shift, data_mask,
                                         cam_fps,
                                         frame_time_ms,
                                         thresh,
                                         thresh,
                                         key_coding=KEY_SPINNAKER)

        spk_img[:] = render_frame(spikes, curr, cam_w, cam_w, polarity)
        cv2.imshow("fig", spk_img)
        cv2.waitKey(10)
        print("frame %s"%img_on_frame)
        t_idx = 0
        for spk_list in lists:

            for spk in spk_list:
                spk_txt = "%s %s"%(spk, t+t_idx)
                write_buff.append(spk_txt)
          
            t_idx += 1
               
        t += frame_time_ms

        # print("img %d, time %s, sim time %s"%(img_idx, prev_ms - start_ms, t))

    ref[:] = 0

    outf = open(spk_fname%img_idx, "w")
    outf.write("\n".join(write_buff))
    outf.close()
    write_buff[:] = []

    if (img_idx + 1)%100 == 0:
      print("MNIST %s set: image %s"%(setname, img_idx+1))


    t += off_time_ms # + frame_time_ms


print("done converting images!!!")

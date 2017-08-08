from __future__ import print_function
import numpy as np
from numpy import int16, uint8
import cv2
from time import time, sleep
from pydvs.virtual_cam import VirtualCam

import pydvs.generate_spikes as gs

import sys
import os
import glob

if len(sys.argv) != 6:
    print("Usage:")
    print("\tpython sequence_to_spikes.py images_dir output_dir  width height frames_per_second\n\n")
    
    sys.exit(0)

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
    imgs = []
    image_list = glob.glob(os.path.join(img_dir, "*.png"))
    image_list.sort()
    for img in image_list:
        if os.path.isfile(img):
            imgs.append(img)

    return imgs


def update_ref(output_type, abs_diff, spikes, ref, thresh, frame_time_ms, \
               num_spikes=1, history_weight=1., log2_table=None):
    
    if output_type == OUTPUT_RATE:
        return gs.update_reference_rate(abs_diff, spikes, ref, thresh,
                                     frame_time_ms,
                                     history_weight)

    elif output_type == OUTPUT_TIME_BIN_THR:
        
        return gs.update_reference_time_binary_thresh(abs_diff, spikes, ref,
                                                   thresh,
                                                   frame_time_ms,
                                                   num_spikes=num_spikes,
                                                   history_weight=history_weight,
                                                   log2_table=log2_table)
    else:
        return gs.update_reference_time_thresh(abs_diff, spikes, ref,
                                            thresh,
                                            frame_time_ms,
                                            history_weight)
    



def make_spikes_lists(output_type, pos, neg, max_diff, \
                      flag_shift, data_shift, data_mask, \
                      frame_time_ms, thresh, \
                      num_bins=1, log2_table=None):

    if output_type == OUTPUT_RATE:

        return gs.make_spike_lists_rate(pos, neg, max_diff,
                                     thresh,
                                     flag_shift, data_shift, data_mask,
                                     frame_time_ms,
                                     key_coding=KEY_SPINNAKER)

    elif output_type == OUTPUT_TIME_BIN_THR:

        return gs.make_spike_lists_time_bin_thr(pos, neg, max_diff,
                                             flag_shift, data_shift, data_mask,
                                             frame_time_ms,
                                             thresh,
                                             thresh,
                                             num_bins,
                                             log2_table,
                                             key_coding=KEY_SPINNAKER)
    else:

        return gs.make_spike_lists_time(pos, neg, max_diff,
                                     flag_shift, data_shift, data_mask,
                                     frame_time_ms,
                                     frame_time_ms,
                                     thresh,
                                     thresh,
                                     key_coding=KEY_SPINNAKER)




orig_w = int(sys.argv[3])
orig_h = int(sys.argv[4])
cam_w = orig_w
cam_h = orig_h
# cam_w = 28
cam_fps = int(sys.argv[5])
frame_time_ms = np.round(1000./cam_fps)

frames_per_image = 1
on_time_ms = frame_time_ms*frames_per_image

img_idx = 1
start_img_idx = 0

num_cycles = 1
frames_per_saccade = cam_fps//3 - 1
frames_per_microsaccade = 1
polarity_name = MERGED_POLARITY
# polarity_name = RECTIFIED_POLARITY
polarity = POLARITY_DICT[polarity_name]

output_type = OUTPUT_RATE
if output_type == OUTPUT_TIME or output_type == OUTPUT_RATE:
    num_bins = np.floor(frame_time_ms)
else:
    num_bins = 8.

t_bin_ms = frame_time_ms//num_bins
print("cam_fps, frame_time_ms, num_bins, t_bin_ms")
print(cam_fps, frame_time_ms, num_bins, t_bin_ms)
print("num_bins, t_bin_ms")
print(num_bins, t_bin_ms)
rate_code = output_type == OUTPUT_RATE
log_time_code = (output_type == OUTPUT_TIME_BIN_THR) or \
                (output_type == OUTPUT_TIME_BIN)
print("using log spike time coding? %s"%(log_time_code))
history_weight = 1.#0.99
behaviour = VirtualCam.BEHAVE_MICROSACCADE
max_dist = 1
horz_bits = uint8(np.ceil(np.log2(cam_w)))
vert_bits = uint8(np.ceil(np.log2(cam_h)))
data_shift = vert_bits
flag_shift = uint8(horz_bits + vert_bits)
data_mask  = uint8(2**horz_bits - 1)

# num_spikes = 1
# log2_table = generate_log2_table(num_spikes, num_bins)[0]
# print("after generate_log2_table")
log2_table = None

inh_width = 2
is_inh_on = False
inh_coords = gs.generate_inh_coords(cam_w, cam_h, inh_width)

print("after generate_inh_coords")

thresh = int( (2**8 - 1)*0.1 )
dx = 1.
#deg = int(sys.argv[1])


orig_img = np.zeros((orig_h, orig_w), dtype=int16)
padd_img = np.zeros((cam_h, cam_w), dtype=int16)
ref_start = 0#32
ref = np.ones((cam_h, cam_w), dtype=int16)*ref_start
curr = np.zeros((cam_h, cam_w), dtype=int16)
diff = np.zeros((cam_h, cam_w), dtype=int16)
abs_diff = np.zeros((cam_h, cam_w), dtype=int16)
spikes = np.zeros((cam_h, cam_w), dtype=int16)
spk_img = np.zeros((cam_h, cam_w, 3), dtype=uint8)
fr_c = (cam_w - orig_w)//2
to_c = fr_c + orig_w
fr_r = (cam_h - orig_h)//2
to_r = fr_r + orig_h

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
prev_ms = time()*1000.
start_ms = time()*1000.
in_dir = sys.argv[1]
image_paths = get_filenames(in_dir)
num_images = len(image_paths)
if num_images == 0:
    print("No images found in path %s"%in_dir)
    sys.exit(0)


out_dir_name = sys.argv[2]
if not os.path.exists(out_dir_name):
    os.makedirs(out_dir_name)
    print("created %s"%out_dir_name)


filelist = glob.glob("%s/*.txt"%(out_dir_name))
for f in filelist:
    os.remove(f)

spk_fname = "spikes_pol_%s_enc_%s_thresh_%d_hist_%d_inh_%s___%d_frames_at_%dfps_%dx%d_res_spikes.txt"%\
            (polarity_name, output_type, thresh, int(history_weight*100), \
             is_inh_on, frames_per_image, cam_fps, cam_w, cam_h)


cx = 0
cy = 0
bg_gray = 0
filename = ""
fade_mask = cv2.imread("./pyDVS/fading_mask.png", cv2.IMREAD_GRAYSCALE)
fade_mask = cv2.resize(fade_mask, (cam_w, cam_w))
fade_mask = np.float64(fade_mask/(255.))
ref[:] = ref_start

for img_idx in range(num_images):
    print(img_idx)
    filename = image_paths[img_idx]
    orig_img[:] = cv2.imread(filename, cv2.IMREAD_GRAYSCALE)
    padd_img[:] = 0
    padd_img[fr_r:to_r, fr_c:to_c] = orig_img 
    curr = padd_img

#     curr *= fade_mask
    
    diff[:], abs_diff[:], spikes[:] = gs.thresholded_difference(curr, ref, thresh)
    
    if is_inh_on:
        spikes[:] = gs.local_inhibition(spikes, abs_diff, inh_coords, 
                                     cam_w, cam_h, inh_width)


    ref[:] = update_ref(output_type, abs_diff, spikes, ref, thresh, frame_time_ms, \
                        num_bins, history_weight, log2_table)

    neg, pos, max_diff = gs.split_spikes(spikes, abs_diff, polarity)

    lists[:] = []
    
    lists[:] = make_spikes_lists(output_type, pos, neg, max_diff,
                                 flag_shift, data_shift, data_mask,
                                 frame_time_ms,
                                 thresh,
                                 num_bins, log2_table)

    spk_img[:] = gs.render_frame(spikes, curr, cam_w, cam_h, polarity)
    cv2.imshow("fig", spk_img)
    key = cv2.waitKey(10) & 0xFF
    if key == ord('q') or key == ord('Q'):
      break
    
    t_idx = 0
    
    for spk_list in lists:
        # print("--------------------------------------------", t_idx)
        for spk in spk_list:
            spk = np.uint16(spk)
            # print(t, t_idx)
            spk_txt = "%u %f"%(spk, t + t_idx)
            # print(spk_txt)
            write_buff.append(spk_txt)
      
        t_idx += t_bin_ms
           
    t += frame_time_ms

    # print("img %d, time %s, sim time %s"%(img_idx, prev_ms - start_ms, t))
    
    # break

outf = open(os.path.join(out_dir_name, spk_fname), "w")
outf.write("\n".join(write_buff))
outf.close()
write_buff[:] = []

# sys.exit(0)

print("done converting images!!!")

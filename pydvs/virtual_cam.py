from __future__ import print_function
import os
import inspect
import glob2
from threading import Thread, Lock
from time import time as get_time
import copy
import numpy as np
from numpy import int16
DTYPE = int16

import cv2

import pydvs.generate_spikes as gs





class VirtualCam():

  BEHAVE_MICROSACCADE = "SACCADE"
  BEHAVE_ATTENTION    = "ATTENTION"
  BEHAVE_TRAVERSE     = "TRAVERSE"
  BEHAVE_FADE         = "FADE"
  BEHAVE_NONE         = "NONE"
  IMAGE_TYPES = ["png", 'jpeg', 'jpg']

  def __init__(self, image_location, behaviour="SACCADE", fps=90, resolution=128,
               image_on_time_ms = 1000, inter_off_time_ms = 100,
               max_saccade_distance=1, frames_per_microsaccade=1, frames_per_saccade=29,
               start_img_idx=0, max_num_images = 60000, max_cycles=3, fade_with_mask=True,
               background_gray=0):

    self.max_cycles = max_cycles
    self.num_cycles = 0
    self.max_num_images = max_num_images
    self.start_img_idx = start_img_idx
    self.total_images = 0
    self.image_filenames = self.get_images_paths(image_location)

    self.fps = fps
    self.time_period = 1./fps

    self.half_frame = int( (fps*(image_on_time_ms/1000.))/2. )

    self.image_on_time = image_on_time_ms/1000.
    self.inter_off_time = inter_off_time_ms/1000.

    self.max_saccade_distance = max_saccade_distance
    self.traverse_speed = (resolution*2.)//(self.image_on_time*self.fps)
    self.frames_per_microsaccade = frames_per_microsaccade
    self.frames_per_saccade = frames_per_saccade
    self.background_gray = background_gray
    self.center_x = 0
    self.center_y = 0

    self.img_smaller = False

    self.img_height = 0#original resolution
    self.img_width  = 0

    self.from_col   = 0
    self.to_col     = 0
    self.from_row   = 0
    self.to_row     = 0

    self.width = resolution #target resolution
    self.height = resolution
    self.shape = (self.height, self.width)
    self.scaled_width = 0

    self.behaviour = behaviour

    self.first_run = True
    self.running = True
    self.locked = False

    self.gray_image = None
    self.tmp_image  = None
    self.tmp_orig = np.zeros(self.shape, dtype=DTYPE)
    self.original_image = np.zeros(self.shape, dtype=DTYPE)
    self.current_image  = np.zeros(self.shape, dtype=DTYPE)

    self.current_image_idx = 0
    self.global_image_idx = 0

    self.frame_prev_time = get_time()

    self.frame_number = 0

    self.current_buffer = 0
    self.buffer_size = 20
    self.half_buffer_size = self.buffer_size//2
    self.all_in_buffer = self.total_images <= self.buffer_size
    self.image_buffer = [ [], [] ]
    if self.all_in_buffer:
      self.buffer_size = self.total_images
      for _ in range(self.buffer_size):
        self.image_buffer[self.current_buffer].append( np.zeros(self.shape, dtype=DTYPE) )
    else:
      for _ in range(self.buffer_size):
        self.image_buffer[self.current_buffer].append( np.zeros(self.shape, dtype=DTYPE) )
        self.image_buffer[int(not self.current_buffer)].append( \
                                          np.zeros(self.shape, dtype=DTYPE) )

    self.buffer_start_idx = 0
    self.load_images(self.current_buffer, self.global_image_idx, self.current_image_idx)

    self.current_image[:] = self.image_buffer[self.current_buffer][0]

    self.fade_with_mask = fade_with_mask
    if fade_with_mask:
        to_per_unit = 1./255.
        dirname = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
        f = os.path.join(dirname, "fading_mask.png")
        self.fade_mask = cv2.imread(f, cv2.IMREAD_GRAYSCALE)*to_per_unit
        

    if self.behaviour == VirtualCam.BEHAVE_FADE or \
       self.behaviour == VirtualCam.BEHAVE_TRAVERSE:
      self.current_image[:] = self.background_gray



    if not self.all_in_buffer:
      self.double_buffer_thread = Thread(name="buffering",
                                         target=self.handle_double_buffering)
      self.double_buffer_thread.start()

    self.on_off_start_time = get_time()
    self.showing_img = True
    #~ self.locking_thread.start()



  def frame_rate_constraint(self, time_period):
    prev_time = get_time()
    lock = Lock()

    while self.running:
      curr_time = get_time()

      if curr_time - prev_time >= time_period:
        lock.acquire()
        try:
          self.locked = False
        finally:
          lock.release()
        #~ print(curr_time - prev_time)
        prev_time = curr_time

    self.locked = False



  def handle_double_buffering(self):

    while self.running:
      curr_idx_copy = copy.copy(self.current_image_idx)
      global_idx_copy = copy.copy(self.global_image_idx)
      curr_buff_copy = copy.copy(self.current_buffer)
      if curr_idx_copy == 1:
        while curr_idx_copy == self.current_image_idx and self.running:
          pass

        self.load_images(int(not curr_buff_copy), global_idx_copy, curr_idx_copy)


  def __del__(self):
    self.stop()

  def stop(self):
    self.running = False

  def isOpened(self):
    return True

  def get(self, prop):
    if prop == cv2.CAP_PROP_FRAME_WIDTH:
      return self.width
    elif prop == cv2.CAP_PROP_FRAME_HEIGHT:
      return self.height
    elif prop == cv2.CAP_PROP_FPS:
      return self.fps
    else:
      return False

  def release(self):
    self.stop()

  def load_images(self, buffer_number, global_idx, curr_idx):
    num_imgs = self.total_images

    # print("in load_images")
    # print("\tglobal idx %s, current idx %s"%(self.global_image_idx, self.current_image_idx))
    # print("\tCOPIES global idx %s, current idx %s"%(global_idx, curr_idx))
    if global_idx == 0 and curr_idx == 0:
      from_idx = 0
    else:
      from_idx = min(global_idx + self.buffer_size - 1, num_imgs)
      from_idx = 0 if from_idx == num_imgs else from_idx
    to_idx = min(from_idx + self.buffer_size, num_imgs)
    # print("\tbuffer %s, from %s to %s"%(buffer_number, from_idx, to_idx))
    buff_idx = 0
    for idx in range(from_idx, to_idx):
    #   print("\tcurrent buffer %s"%(self.current_buffer))
    #   print("\tfilling buffer %s, buff_idx %s <-> img idx %s"%(buffer_number, buff_idx, idx))

      self.image_buffer[buffer_number][buff_idx][:] = self.grab_image(idx)
      buff_idx += 1





  def read(self, ref):
    image_on_time = self.image_on_time
    inter_off_time = self.inter_off_time
    background = self.background_gray
    behaviour = self.behaviour
    traverse = VirtualCam.BEHAVE_TRAVERSE
    fade = VirtualCam.BEHAVE_FADE
    num_images = self.total_images
    all_in_buffer = self.all_in_buffer
    buffer_size = self.buffer_size

    start = get_time()
    run_time = start - self.on_off_start_time
    if behaviour == VirtualCam.BEHAVE_NONE:
      self.current_image = self.image_buffer[self.current_buffer]\
                                            [self.current_image_idx]
      self.current_image_idx += 1
      self.global_image_idx += 1

      if self.global_image_idx == num_images:
        self.current_image_idx = 0
        self.global_image_idx  = 0
        if not all_in_buffer:
          self.current_buffer = int(not self.current_buffer)

      elif self.current_image_idx == buffer_size:
        self.current_image_idx = 0
        if not all_in_buffer:
          self.current_buffer = int(not self.current_buffer)



    else:
      if self.num_cycles == self.max_cycles:
        self.current_image[:] = background
        self.original_image[:] = background
        return False, self.current_image

      if not self.showing_img:

        if run_time >= inter_off_time:
          self.showing_img = True
          self.on_off_start_time = get_time()

        #   self.current_image[:] = self.image_buffer[self.current_buffer][self.current_image_idx]
        #   self.original_image[:] = self.current_image
        #   print("buffer %s, index %s, frame %s, global idx %s"%\
        #         (self.current_buffer, self.current_image_idx, self.frame_number,
        #          self.global_image_idx))


          self.current_image_idx += 1
          self.global_image_idx += 1

          if self.global_image_idx == num_images:
            self.num_cycles += 1
            if self.num_cycles == self.max_cycles:
                self.current_image[:] = background
                self.original_image[:] = background
                return False, self.current_image
            self.current_image_idx = 0
            self.global_image_idx  = 0
            if not all_in_buffer:
              self.current_buffer = int(not self.current_buffer)

          elif self.current_image_idx == buffer_size:
            self.current_image_idx = 0
            if not all_in_buffer:
              self.current_buffer = int(not self.current_buffer)

          self.frame_number = 0

          self.original_image[:] = self.image_buffer[self.current_buffer][self.current_image_idx]
          self.current_image[:] = self.original_image

          if behaviour == traverse or behaviour == fade:
            self.current_image[:] = background
          else:
            self.center_x = 0
            self.center_y = 0

        else:
          self.current_image[:] = background

      else:
        if run_time >= image_on_time:
          self.showing_img = False
          self.on_off_start_time = get_time()
          self.frame_number = 0
        else:

          self.move_image(ref)
          if self.fade_with_mask:
              self.current_image[:] = gs.mask_image(self.current_image,
                                                 self.fade_mask)
      self.frame_number += 1



      self.prev_time = get_time()

    return True, self.current_image


  def grab_image(self, idx):

    first_run = self.first_run
    filename = self.image_filenames[idx]
    width = self.width; height = self.height
    # original = self.original_image
    original = self.tmp_orig
    if first_run:
      self.gray_image = cv2.imread(filename, cv2.IMREAD_GRAYSCALE)
      gray = self.gray_image

      if gray is None:
        print("could not read image %s"%(filename))
        original[:] = 0

      self.img_height, self.img_width = gray.shape
      img_width = self.img_width; img_height = self.img_height

      if img_width <= width and img_height <= height:
        self.img_smaller = True

        diff = width - img_width
        from_col = diff//2
        to_col = from_col + img_width

        diff = height - img_height
        from_row = diff//2
        to_row = from_row + img_height

        original[from_row:to_row, from_col:to_col] = gray

      else:

        self.scaled_width = int(float(img_width*height)/img_height)
        scaled_width = self.scaled_width
        print(img_width, width, scaled_width)

        diff = scaled_width - width
        from_col = diff//2
        to_col = from_col + width
        from_row = 0
        to_row = 0
        self.tmp_image = cv2.resize(gray, (scaled_width, height), interpolation=cv2.INTER_NEAREST)

        original[:] = self.tmp_image[:, from_col:to_col]

      self.first_run = False
      self.from_col = from_col; self.to_col = to_col
      self.from_row = from_row; self.to_row = to_row

    else:
      gray = self.gray_image
      img_smaller = self.img_smaller
      from_col = self.from_col; to_col = self.to_col
      from_row = self.from_row; to_row = self.to_row
      scaled_width = self.scaled_width

      gray[:] = cv2.imread(filename, cv2.IMREAD_GRAYSCALE)

      if img_smaller:
        original[from_row:to_row, from_col:to_col] = gray

      else:
        self.tmp_image[:] = cv2.resize(gray, (scaled_width, height), interpolation=cv2.INTER_NEAREST)
        original[:] = self.tmp_image[:, from_col:to_col]

    return original.copy()



  def move_image(self, ref):
    img = self.current_image
    orig = self.original_image
    behaviour = self.behaviour
    half_frame = self.half_frame
    max_dist = self.max_saccade_distance
    speed = self.traverse_speed
    fp_uscd = self.frames_per_microsaccade
    fp_scd  = self.frames_per_saccade
    bg_gray = self.background_gray
    cx = self.center_x
    cy = self.center_y
    frame_number = self.frame_number


    if behaviour == VirtualCam.BEHAVE_TRAVERSE:
        img[:] = gs.traverse_image(orig, frame_number, speed, bg_gray)

    elif behaviour == VirtualCam.BEHAVE_FADE:
        img[:] = gs.fade_image(orig, frame_number, half_frame, bg_gray)

    elif behaviour == VirtualCam.BEHAVE_MICROSACCADE:
        img[:], cx, cy = gs.usaccade_image(orig, frame_number, fp_uscd,
                                        max_dist, cx, cy, bg_gray)

    else:
        img[:], cx, cy = gs.attention_image(orig, img, ref, frame_number,
                                         fp_uscd, fp_scd,
                                         max_dist, cx, cy, bg_gray)

    self.center_x = cx
    self.center_y = cy

    return img


  def get_images_paths(self, images):

    imgs = []
    self.total_images = 0
    if type(images) == type(list()): #if we've got a list of image paths
      for img in images:
        if os.path.isfile(img): #check if the image file exists
          imgs.append(img)
          self.total_images += 1

    elif type(images) == type(str()): # if we get a string

      if os.path.isfile(images):  # is it a file?
        imgs.append(images)
        self.total_images += 1

      elif os.path.isdir(images): # or a directory?
        img_idx = 0
        for extension in self.IMAGE_TYPES:
          image_list = glob2.glob(os.path.join(images, "*.%s"%extension))
          if image_list is None:
            continue
          image_list.sort()
          for img in image_list:
            if os.path.isfile(img):

              if img_idx < self.start_img_idx:
                img_idx += 1
                continue
              else:
                imgs.append(img)
                self.total_images += 1

            if self.total_images == self.max_num_images:
                return imgs
            img_idx += 1


    if len(imgs) == 0:
        raise Exception("No images loaded! ")

    for img in imgs:
      print(img)

    return imgs

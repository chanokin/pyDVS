import cv2
import numpy as np
try:
    from cv2.cv import CV_INTER_AREA
except:
    from cv2 import INTER_AREA as CV_INTER_AREA

DTYPE = np.float32
DTYPE_str = 'h'
DTYPE_KEY = np.int32
DTYPE_IDX = np.int64
DTYPE_U8 = np.uint8
DTYPE_U16 = np.uint16
DTYPE_FLOAT = np.float

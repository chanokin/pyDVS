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

MODE_128 = "128"
MODE_64  = "64"
MODE_32  = "32"
MODE_16  = "16"

UP_POLARITY     = "UP"
DOWN_POLARITY   = "DOWN"
MERGED_POLARITY = "MERGED"
# POLARITY_DICT   = {
#     UP_POLARITY: np.uint8(0), 
#     DOWN_POLARITY: np.uint8(1), 
#     MERGED_POLARITY: np.uint8(2),
#     np.uint8(0): UP_POLARITY,
#     np.uint8(1): DOWN_POLARITY,
#     np.uint8(2): MERGED_POLARITY
# }

OUTPUT_RATE         = "RATE"
OUTPUT_TIME         = "TIME"
OUTPUT_TIME_BIN     = "TIME_BIN"
OUTPUT_TIME_BIN_THR = "TIME_BIN_THR"

BEHAVE_MICROSACCADE = "SACCADE"
BEHAVE_ATTENTION    = "ATTENTION"
BEHAVE_TRAVERSE     = "TRAVERSE"
BEHAVE_FADE         = "FADE"

IMAGE_TYPES = ["png", 'jpeg', 'jpg']
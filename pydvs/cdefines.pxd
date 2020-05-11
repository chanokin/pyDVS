import numpy as np
cimport numpy as np
cimport cython

ctypedef np.float32_t DTYPE_t
ctypedef np.int32_t DTYPE_KEY_t
ctypedef np.int64_t DTYPE_IDX_t
ctypedef np.int64_t DTYPE_I64_t
ctypedef np.uint8_t DTYPE_U8_t
ctypedef np.uint16_t DTYPE_U16_t
ctypedef np.float_t DTYPE_FLOAT_t

DEF UP_POLARITY = 0
DEF DOWN_POLARITY = 1
DEF MERGED_POLARITY = 2
DEF RECTIFIED_POLARITY = 3
DEF VALS = 2
DEF COLS = 1
DEF ROWS = 0

DEF KEY_SPINNAKER = 0
DEF KEY_XYP = 1

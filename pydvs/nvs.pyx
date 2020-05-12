from cython.parallel import prange
import array
from cpython cimport array
import time
import cv2
import numpy as np
cimport numpy as np

cimport cython

from pydvs.cdefines cimport *
from pydvs.pdefines import *

from pydvs.math_utils cimport *
from pydvs.math_utils_np import *

cimport pydvs.generate_spikes as gen

cdef class NVSEmu(object):
    cdef dict _config
    cdef DTYPE_IDX_t _width, _height
    cdef DTYPE_t _rec_dec
    cdef DTYPE_t _thr_inc, _thr_dec, _thr0
    cdef DTYPE_t _vmin, _vmax

    cdef DTYPE_t[:] _ref_on, _ref_off
    cdef DTYPE_t[:] _thr_on, _thr_off
    cdef DTYPE_t[:] _spk_on, _spk_off, _spk_out

    def __init__(self, tuple shape, dict config=NVSEmu.default_config):
        self._width, self._height = shape
        self._config = config
        self._config_parameters(config)
        self._init_buffers(shape)
    
    cdef _init_buffers(self, tuple shape):
        self._ref_on = np.ones(shape, dtype=DTYPE) * self._vmin
        self._ref_off = np.ones(shape, dtype=DTYPE) * self._vmax

        self._thr_on = np.ones(shape, dtype=DTYPE) * self._thr0
        self._thr_off = np.ones(shape, dtype=DTYPE) * self._thr0

        self._spk_on = np.zeros(shape, dtype=DTYPE) 
        self._spk_off = np.zeros(shape, dtype=DTYPE)
        self._spk_out = np.zeros(shape, dtype=DTYPE)

    cdef _config_parameters(self, config):
        self._ref_dec = self._config_reference(config)
        self._thr_inc, self._thr_dec, self._thr0 = self._config_threshold(config)
        self._vmin, self._vmax = self._config_ranges(config)

    cdef _config_reference(self, config):
        dc = NVSEmu.default_config
        return config.get('reference', dc['reference'])['dec']

    cdef _config_threshold(self, config):
        dc = NVSEmu.default_config
        thr_inc = config.get('threshold', dc['threshold'])['increment']
        thr_dec = config.get('threshold', dc['threshold'])['dec']
        thr0 = config.get('threshold', dc['threshold'])['base']
        
        return thr_inc, thr_dec, thr0

    cdef _config_ranges(self, config):
        dc = NVSEmu.default_config
        vmin = config.get('range', dc['range'])['minimum']
        vmax = config.get('range', dc['range'])['maximum']
        
        return vmin, vmax

    cpdef update(self, np.ndarray[DTYPE_t, ndim=2] input_image):
        cdef np.ndarray[DTYPE_t, ndim=2] diff, abs_diff_on, abs_diff_off

        diff, abs_diff_on, self._spk_on = \
            gen.thresholded_difference(input_image, self._ref_on, self._thr_on)
        diff, abs_diff_off, self._spk_off = \
            gen.thresholded_difference(input_image, self._ref_off, self._thr_off)
        
        self._ref_on = gen.update_reference(
                        self._ref_on, self._spk_on, self._thr_on, 
                        self._ref_dec, self._vmin, self._vmin, self._vmax)
        self._ref_off = gen.update_reference(
                            self._ref_off, self._spk_off, self._thr_off, 
                            self._ref_dec, self._vmax, self._vmin, self._vmax)

        self._thr_on = gen.update_threshold(
                        self._thr_on, self._spk_on,
                        self._thr_inc, self._thr_dec, self.thr0)
        self._thr_off = gen.update_threshold(
                        self._thr_off, self._spk_off,
                        self._thr_inc, self._thr_dec, self.thr0)

        self._spk_out = gen.get_output_spikes(
                            abs_on, abs_off, self._spk_on, self._spk_off)

    cdef dict default_config(self):
        return  {
                    'reference': {
                        'dec': DTYPE( np.exp(-1.0 / 40.0) ), # frames
                    },
                    'threshold':{
                        'dec': DTYPE( np.exp(-1.0 / 5.0) ), # frames
                        'increment': DTYPE( 1.3 ), # mult
                        'base': DTYPE( 0.05 * 255.0 ), # v
                    },
                    'range':{
                        'minimum': DTYPE( 0.0 ), # v
                        'maximum': DTYPE( 255.0 ), # v
                    },
                }

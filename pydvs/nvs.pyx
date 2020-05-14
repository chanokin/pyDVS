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


_default_config = {
    'reference': {
        'dec': DTYPE( np.exp(-1.0 / 40.0) ), # frames
    },
    'threshold':{
        'dec': DTYPE( np.exp(-1.0 / 50.0) ), # frames
        'increment': DTYPE( 1.8 ), # mult
        'base': DTYPE( 0.10 * 255.0 ), # v
    },
    'range':{
        'minimum': DTYPE( 0.0 ), # v
        'maximum': DTYPE( 255.0 ), # v
    },
}


cdef class NVSEmu(object):
    cdef bint _scale
    cdef DTYPE_IDX_t _width, _height
    cdef DTYPE_t _ref_dec
    cdef DTYPE_t _thr_inc, _thr_dec, _thr0
    cdef DTYPE_t _vmin, _vmax

    cdef DTYPE_t[:, :] _ref_on, _ref_off
    cdef DTYPE_t[:, :] _thr_on, _thr_off
    cdef DTYPE_t[:, :] _spk_on, _spk_off, _spk_out_on, _spk_out_off

    def __init__(self, tuple shape, bint scale, dict config=_default_config):
        self._height, self._width = shape
        self._config_parameters(config)
        self._init_buffers(shape)
        self._scale = scale
    
    def spikes(self):
        return self._spk_out_on, self._spk_out_off
    
    cdef _init_buffers(self, tuple shape):
        self._ref_on = np.ones(shape, dtype=DTYPE) * self._vmin
        self._ref_off = np.ones(shape, dtype=DTYPE) * self._vmax

        self._thr_on = np.ones(shape, dtype=DTYPE) * self._thr0
        self._thr_off = np.ones(shape, dtype=DTYPE) * self._thr0

        self._spk_on = np.zeros(shape, dtype=DTYPE) 
        self._spk_off = np.zeros(shape, dtype=DTYPE)

        self._spk_out_on = np.zeros(shape, dtype=DTYPE)
        self._spk_out_off = np.zeros(shape, dtype=DTYPE)

    cdef _config_parameters(self, config):
        self._ref_dec = self._config_reference(config)
        self._thr_inc, self._thr_dec, self._thr0 = self._config_threshold(config)
        self._vmin, self._vmax = self._config_ranges(config)

    cdef _config_reference(self, config):
        dc = _default_config
        return config.get('reference', dc['reference'])['dec']

    cdef _config_threshold(self, config):
        dc = _default_config
        thr_inc = config.get('threshold', dc['threshold'])['increment']
        thr_dec = config.get('threshold', dc['threshold'])['dec']
        thr0 = config.get('threshold', dc['threshold'])['base']
        
        return thr_inc, thr_dec, thr0

    cdef _config_ranges(self, config):
        dc = _default_config
        vmin = config.get('range', dc['range'])['minimum']
        vmax = config.get('range', dc['range'])['maximum']
        
        return vmin, vmax

    cpdef update(self, np.ndarray[DTYPE_t, ndim=2] input_image):
        cdef np.ndarray[DTYPE_t, ndim=2] scl, abs_diff_on, abs_diff_off

        scl = np.zeros((self._height, self._width), dtype=DTYPE)
        abs_diff_on = np.zeros((self._height, self._width), dtype=DTYPE)
        abs_diff_off = np.zeros((self._height, self._width), dtype=DTYPE)

        if self._scale:
            scl = cv2.resize(
                    input_image, (self._width, self._height), interpolation=INTER_LINEAR)
        else:
            scl = input_image

        abs_diff_on[:], self._spk_on = \
            gen.thresholded_difference(scl, self._ref_on, self._thr_on)
        
        self._ref_on = gen.update_reference(
                        self._ref_on, self._spk_on, self._thr_on, 
                        self._ref_dec, self._vmin, self._vmin, self._vmax)

        self._thr_on = gen.update_threshold(
                        self._thr_on, self._spk_on,
                        self._thr_inc, self._thr_dec, self._thr0)

        abs_diff_off[:], self._spk_off = \
            gen.thresholded_difference(scl, self._ref_off, self._thr_off)

        self._ref_off = gen.update_reference(
                            self._ref_off, self._spk_off, self._thr_off, 
                            self._ref_dec, self._vmax, self._vmin, self._vmax)

        self._thr_off = gen.update_threshold(
                        self._thr_off, self._spk_off,
                        self._thr_inc, self._thr_dec, self._thr0)

        self._spk_out_on, self._spk_out_off = \
            gen.get_output_spikes(
                abs_diff_on, abs_diff_off, self._spk_on, self._spk_off)



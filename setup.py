'''
Created on Nov 24, 2009
Call python setup.py build_ext --inplace
@author: fred
'''
import numpy
from distutils.core import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext

ext_modules = [
                Extension("pydvs.decode_spikes", ["pydvs/decode_spikes.pyx"], include_dirs=[numpy.get_include()]),
                Extension("pydvs.generate_spikes_uncoded", ["pydvs/generate_spikes_uncoded.pyx"], include_dirs=[numpy.get_include()]),
                Extension("pydvs.generate_spikes", ["pydvs/generate_spikes.pyx"], include_dirs=[numpy.get_include()])
               ]

setup(
  name = 'Spike generators',
  cmdclass = {'build_ext': build_ext},
  ext_modules = ext_modules
)

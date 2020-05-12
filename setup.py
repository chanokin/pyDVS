'''
Created on Nov 24, 2009
Call python setup.py build_ext --inplace
@author: fred
'''
import numpy
from distutils.core import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext
from Cython.Build import cythonize
import glob
import os

def to_pack(fname, pack):
    mod = os.path.basename(fname)[:-4] # remove .pyx from fname
    packed = '{}.{}'.format(pack, mod)
    return packed

pack_name = "pydvs"
base = os.path.dirname(os.path.realpath(__file__))
mod_dir = os.path.join(base, pack_name)
cython_files = glob.glob(os.path.join(mod_dir, "*.pyx"))

ext_modules = [
    Extension(to_pack(f, pack_name), [f], 
              include_dirs=[numpy.get_include()],
              define_macros=[("NPY_NO_DEPRECATED_API", "NPY_1_7_API_VERSION")],)
    for f in cython_files
]


setup(
    name='NVS Emulator',
    ext_modules=cythonize(ext_modules, 
                          include_path=[numpy.get_include()],
                          language='c++',
                          language_level='3')
)

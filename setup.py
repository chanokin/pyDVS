'''
Created on Nov 24, 2009
Call python setup.py build_ext --inplace
@author: fred
'''
import numpy
from distutils.core import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext
import glob
import os

def to_pack(fname, pack):
    mod = os.path.basename(fname)[:-4]
    packed = '{}.{}'.format(pack, mod)
    return packed

pack_name = "pydvs"
base = os.path.dirname(os.path.realpath(__file__))
mod_dir = os.path.join(base, pack_name)
cython_files = glob.glob(os.path.join(mod_dir, "*.pyx"))

ext_modules = [
    Extension(to_pack(f, pack_name), [f], include_dirs=[numpy.get_include()])
    for f in cython_files
]

# this is now mandatory, explicitly state Python 3
for e in ext_modules:
    e.cython_directives = {'language_level': "3"}

setup(
    name='Spike generators',
    cmdclass={'build_ext': build_ext},
    ext_modules=ext_modules
)

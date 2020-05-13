'''
Created on Nov 24, 2009
Call python setup.py build_ext --inplace
@author: fred
'''
import numpy
from setuptools import setup, find_packages, Extension
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
    name='PyDVS: NVS Emulator',
    version='0.2.0',
    packages=find_packages(),
    ext_modules=cythonize(ext_modules, 
                          include_path=[numpy.get_include()],
                          language='c++',
                          language_level='3'),
    url='https://github.com/chanokin/pyDVS',
    author='Garibaldi Pineda Garcia and Fred Rotbart',
    description="Cython-based Neuromorphic Vision Sensor emulator",
    install_requires=["cython", "numpy", "cv2",],
    zip_safe=False,  
    
)

from distutils.core import setup
from Cython.Build import cythonize
import numpy

setup(
  name = 'monotonic_alignment_search',
  ext_modules = cythonize("core.pyx"),
  include_dirs=[numpy.get_include()]
)
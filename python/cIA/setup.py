from distutils.core import setup
from Cython.Build import cythonize
import os
import numpy


setup(
    name="cIA",
    description='Cythonized methods for pyIA',
    author='Daniel Dugas',
    version='1.0',
    py_modules=[],
    ext_modules = cythonize("cia.pyx", annotate=True),
    include_dirs=[numpy.get_include()],
)

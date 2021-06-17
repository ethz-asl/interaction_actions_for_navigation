from setuptools import setup, find_packages
from Cython.Build import cythonize
import os
import numpy



setup(
    name="pyIAN",
    description='Python library for IAN multibehavior planner',
    author='Daniel Dugas',
    version='0.0.3',
    packages=find_packages(),
    ext_modules = cythonize("cIA/cia.pyx", annotate=True),
    include_dirs=[numpy.get_include()],
    #     ext_modules = cythonize("clib_clustering/lidar_clustering.pyx", annotate=True),
    package_data={'pyIA': ['maps/*', 'scenarios/*']},
)

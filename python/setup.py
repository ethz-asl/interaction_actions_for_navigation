from setuptools import setup, find_packages
from Cython.Build import cythonize
import os



setup(
    name="pyIA",
    description='Python library for interaction-actions planner',
    author='Daniel Dugas',
    version='0.0',
    packages=find_packages(),
#     ext_modules = cythonize("clib_clustering/lidar_clustering.pyx", annotate=True),
)

# install cIA separately by running pip install . in the cIA subdir!

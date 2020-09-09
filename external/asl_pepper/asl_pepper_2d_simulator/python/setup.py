from setuptools import setup, find_packages
import os

setup(
    name="pepper_2d_simulator",
    description='Python library for Pepper 2d simulation',
    author='Daniel Dugas',
    version='0.0',
#     packages=find_packages(),
    py_modules=[
        'gymified_pepper_envs',
        'pepper_2d_simulator',
        'pepper_2d_iarlenv',
                ],
)


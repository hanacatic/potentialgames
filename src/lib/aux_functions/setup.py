from setuptools import setup
from Cython.Build import cythonize

setup(
    name='aux_functions',
    ext_modules=cythonize("helpers.py"),
)
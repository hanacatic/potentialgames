from setuptools import setup
from Cython.Build import cythonize

setup(
    name='Player',
    ext_modules=cythonize("player.py"),
)
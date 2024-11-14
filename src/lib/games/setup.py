from setuptools import setup
from Cython.Build import cythonize

setup(
    name='games',
    ext_modules=cythonize("*.py"),
)
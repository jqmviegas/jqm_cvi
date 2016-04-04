"""A setuptools based setup module.
See:
https://packaging.python.org/en/latest/distributing.html
https://github.com/pypa/sampleproject
"""

from distutils.core import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext
import numpy

setup(
    name="jqmcvi",

    version="1.0",

    author="Joaquim L. Viegas",
    author_email = "jqmviegas@gmail.com",

    license="MIT",

    cmdclass = {'build_ext': build_ext},
    ext_modules = [
        Extension("jqmcvi.basec", ["jqmcvi/basec.pyx"],
                  include_dirs=[numpy.get_include()]),
    ],
    packages=["jqmcvi"]
)

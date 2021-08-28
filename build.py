import os

from Cython.Build import cythonize
from distutils.core import Extension


def build(setup_kwargs):
    cy_ext = cythonize(
        Extension(
            name="textnets._ext",
            sources=["textnets/_ext.pyx"],
        ),
    )
    setup_kwargs.update({"ext_modules": cy_ext})

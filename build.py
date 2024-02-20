import os
from distutils.core import Extension

from Cython.Build import cythonize


def build(setup_kwargs):
    cy_ext = cythonize(
        Extension(
            name="textnets._ext",
            sources=["textnets/_ext.pyx"],
        ),
        compiler_directives={"language_level": 3},
    )
    setup_kwargs.update({"ext_modules": cy_ext})

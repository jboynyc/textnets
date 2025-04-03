"""
Build hooks for textnets.

Documentation: https://backend.pdm-project.org/hooks/
"""

from Cython.Build import cythonize


def pdm_build_update_setup_kwargs(_, setup_kwargs):
    """Add cythonized extension to pdm-backend generated setup.py."""
    cy_ext = cythonize(["textnets/_ext.pyx"], compiler_directives={"language_level": 3})
    setup_kwargs.update({"packages": ["textnets"], "ext_modules": cy_ext})

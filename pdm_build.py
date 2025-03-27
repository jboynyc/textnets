from Cython.Build import cythonize


def pdm_build_update_setup_kwargs(_, setup_kwargs):
    cy_ext = cythonize(["textnets/_ext.pyx"], compiler_directives={"language_level": 3})
    setup_kwargs.update({"packages": ["textnets"], "ext_modules": cy_ext})

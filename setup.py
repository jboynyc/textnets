#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""The setup script."""

from setuptools import setup, find_packages

with open("README.rst") as readme_file:
    readme = readme_file.read()

requirements = [
    "pandas==1.2.3",
    "cairocffi==1.2.0",
    "python-igraph==0.9.1",
    "spacy==3.0.5",
    "scipy==1.6.1",
    "toolz==0.11.1",
    "leidenalg==0.8.2",
]

test_requirements = [
    "pytest>=4.6.5",
    "pandas",
    "python-igraph",
    "spacy",
    "scipy",
    "toolz",
    "leidenalg",
]

dev_requirements = [
    "pip==20.2.3",
    "bump2version==1.0.1",
    "watchdog==2.1.1",
    "flake8==3.9.2",
    "tox==3.23.1",
    "coverage==5.5",
    "twine==3.4.1",
    "pytest==6.2.4",
    "pytest-runner==5.3.0",
    "black==21.5b1",
    "mypy==0.812",
]

setup(
    author="John D. Boy",
    author_email="jboy@bius.moe",
    python_requires=">=3.7",
    name="textnets",
    version="0.4.11",
    description="Automated text analysis with networks",
    long_description=readme,
    url="https://textnets.readthedocs.io",
    packages=find_packages(include=["textnets", "textnets.*"]),
    include_package_data=True,
    license="GNU General Public License v3",
    zip_safe=False,
    keywords="textnets",
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: GNU General Public License v3 (GPLv3)",
        "Natural Language :: English",
        "Programming Language :: Python :: 3 :: Only",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Topic :: Scientific/Engineering :: Visualization",
        "Topic :: Sociology",
    ],
    test_suite="tests",
    install_requires=requirements,
    extras_require={
        ":python_version<'3.8'": ["typing_extensions", "cached-property"],
        "test": test_requirements,
        "dev": dev_requirements,
        "doc": [
            "Sphinx>4",
            "sphinx_rtd_theme",
            "jupyter_sphinx",
            "sphinxcontrib-bibtex",
        ],
    },
)

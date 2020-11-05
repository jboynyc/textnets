#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""The setup script."""

from setuptools import setup, find_packages

with open("README.rst") as readme_file:
    readme = readme_file.read()

requirements = [
    "Click>=7.0",
    "pandas==1.1.4",
    "cairocffi==1.1.0",
    "python-igraph==0.8.3",
    "spacy==2.3.2",
    "scipy==1.5.4",
    "toolz==0.10.0",
    "leidenalg==0.8.2",
]

test_requirements = [
    "pytest>=4.6.5",
    "Click",
    "pandas",
    "python-igraph",
    "spacy",
    "scipy",
    "toolz",
    "leidenalg",
]

dev_requirements = [
    "pip==19.2.3",
    "bump2version==1.0.0",
    "wheel==0.34.2",
    "watchdog==0.10.3",
    "flake8==3.8.3",
    "tox==3.19.0",
    "coverage==5.2.1",
    "twine==3.2.0",
    "pytest==5.4.3",
    "pytest-runner==5.2",
    "black==19.10b0",
    "mypy==0.782",
]

setup(
    author="John D. Boy",
    author_email="jboy@bius.moe",
    python_requires=">=3.7",
    name="textnets",
    version="0.4.10",
    description="Automated text analysis with networks",
    long_description=readme,
    url="https://textnets.readthedocs.io",
    packages=find_packages(include=["textnets", "textnets.*"]),
    entry_points={"console_scripts": ["textnets=textnets.cli:main"]},
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
            "Sphinx>=3.0.4",
            "sphinx_rtd_theme",
            "jupyter_sphinx",
            "sphinxcontrib-bibtex",
        ],
    },
)

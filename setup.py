#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""The setup script."""

from setuptools import setup, find_packages

with open('README.rst') as readme_file:
    readme = readme_file.read()

requirements = [
    'Click>=7.0',
    'pandas==1.0.3',
    'python-igraph==0.8.2',
    'spacy==2.2.4',
    'toolz==0.10.0',
    'leidenalg==0.8.0'
]

setup_requirements = [
    'pytest-runner',
]

test_requirements = [
    'pytest',
    'Click',
    'pandas',
    'python-igraph',
    'spacy',
    'toolz',
    'leidenalg'
]

setup(
    name='textnets',
    version='0.3.0',
    description="Automated text analysis with networks",
    long_description=readme,
    author="John D. Boy",
    author_email='jboy@bius.moe',
    url='https://github.com/jboynyc/textnets',
    packages=find_packages(include=['textnets']),
    entry_points={
        'console_scripts': [
            'textnets=textnets.cli:main'
        ]
    },
    include_package_data=True,
    install_requires=requirements,
    license="GNU General Public License v3",
    zip_safe=False,
    keywords='textnets',
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: GNU General Public License v3 (GPLv3)',
        'Natural Language :: English',
        'Programming Language :: Python :: 3 :: Only',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Topic :: Scientific/Engineering :: Visualization',
        'Topic :: Sociology',
    ],
    test_suite='tests',
    tests_require=test_requirements,
    setup_requires=setup_requirements,
)

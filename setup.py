#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""The setup script."""

from setuptools import setup, find_packages

with open('README.rst') as readme_file:
    readme = readme_file.read()

requirements = [
    'Click>=6.0',
    'pandas==0.25.3',
    'python-igraph==0.7.1.post6',
    'spacy==2.2.3',
    'toolz==0.10.0',
    'leidenalg==0.7.0'
]

setup_requirements = [
    'pytest-runner',
    # TODO(jboynyc): put setup requirements (distutils extensions, etc.) here
]

test_requirements = [
    'pytest',
    # TODO: put package test requirements here
]

setup(
    name='textnets',
    version='0.1.0',
    description="Automated text analysis with networks.",
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
        'Development Status :: 2 - Pre-Alpha',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: GNU General Public License v3 (GPLv3)',
        'Natural Language :: English',
        "Programming Language :: Python :: 2",
        'Programming Language :: Python :: 2.6',
        'Programming Language :: Python :: 2.7',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.3',
        'Programming Language :: Python :: 3.4',
        'Programming Language :: Python :: 3.5',
    ],
    test_suite='tests',
    tests_require=test_requirements,
    setup_requires=setup_requirements,
)

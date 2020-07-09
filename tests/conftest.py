#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Configure tests for `textnets` package."""

import pytest
from textnets import Corpus, examples


@pytest.fixture(scope="session")
def testdata():
    return examples.moon_landing


@pytest.fixture(scope="session")
def corpus(testdata):
    return Corpus(testdata)

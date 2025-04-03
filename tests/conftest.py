#!/usr/bin/env python

"""Configure tests for `textnets` package."""

import pytest

from textnets import Corpus, examples


@pytest.fixture(scope="session")
def testdata():
    """Provide test data."""
    return examples.moon_landing


@pytest.fixture(scope="session")
def corpus(testdata):
    """Provide a test corpus."""
    return Corpus(testdata)

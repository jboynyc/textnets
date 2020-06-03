.. highlight:: shell

============
Installation
============

**textnets** is in the `Python Package Index`_, so it can be installed using `pip`_.

.. _`Python Package Index`: https://pypi.org/project/textnets/
.. _pip: https://pip.pypa.io

.. note::

   Please note that **textnets** currently requires Python 3.8 to run. There's
   `an issue <https://github.com/jboynyc/textnets/issues/8>`_ open about that.

In a `virtual environment`_, run::

   $ pip install textnets

.. _`virtual environment`: https://packaging.python.org/tutorials/installing-packages/#creating-virtual-environments

This is the preferred method to install **textnets**, as it will always install the most recent stable release.

If you don't have pip installed, the `Python installation guide`_ can guide
you through the process.

.. _Python installation guide: http://docs.python-guide.org/en/latest/starting/installation/

Most likely you will also have to install an appropriate `language model`_ by issuing a command like::

   $ python -m spacy download en_core_web_sm

.. _`language model`: https://spacy.io/usage/models#download

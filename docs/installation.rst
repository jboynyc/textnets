.. highlight:: shell

============
Installation
============

**textnets** is included in `conda-forge`_ and the `Python Package Index`_, so
it can either be installed using `conda`_ or `pip`_.

.. _conda-forge: https://anaconda.org/conda-forge/textnets/
.. _`Python Package Index`: https://pypi.org/project/textnets/
.. _conda: https://conda.io/
.. _pip: https://pip.pypa.io

.. note::

   Please note that **textnets** requires Python 3.8 or newer to run.

Using conda
-----------

This is the preferred method for most users. The `Anaconda Python
distribution`_ is an easy way to get up and running with Python, especially if
you are on a Mac or Windows system.

.. _Anaconda Python distribution: https://www.anaconda.com/products/individual

Once it is installed you can use its package manager ``conda`` to install
**textnets**::

   $ conda install -c conda-forge textnets

This tells conda to install **textnets** from the conda-forge channel.

If you don't know how to enter this command, you can use the Anaconda Navigator
instead. It provides a graphical interface that allows you to install new
packages.

.. admonition:: Installing **textnets** in Anaconda Navigator

   1. Go to the **Environments** tab.
   2. Click the **Channels** button.
   3. Click the **Add** button.
   4. Enter the channel URL https://conda.anaconda.org/conda-forge/
   5. Hit your keyboard's **Enter** key.
   6. Click the **Update channels** button.
   7. Now you can install **textnets** in a new environment. (Make sure the
      package filter on the **Environments** tab is set to "all.")

Using pip
---------

Alternately, if you already have Python installed, you can use its package
manger to install **textnets**. In a `virtual environment`_, run::

   $ python -m pip install textnets

.. _`virtual environment`: https://packaging.python.org/tutorials/installing-packages/#creating-virtual-environments

Plotting
--------

.. sidebar::

    In rare cases you may have to `install CFFI`_ separately for plotting to
    work.

.. _install CFFI: https://cffi.readthedocs.io/en/latest/installation.html

**textnets** installs the `Cairo`_ graphics library as a dependency for
plotting. If you are using a Mac without Anaconda, you will probably have to
install Cairo separately using the `Homebrew`_ package manager.

.. _Cairo: https://www.cairographics.org/
.. _Homebrew: https://formulae.brew.sh/formula/cairo

Language Support
----------------

**textnets** can try to download the `language models`_ you need "on the fly"
if you set the ``autodownload`` parameter to ``True``. (It is off by default
because language models are frequently many hundreds of megabytes in size and
probably shouldn't be downloaded on a metered connection.)

>>> import textnets as tn
>>> tn.params["autodownload"] = True

You can also install the models manually by issuing a command like::

   $ python -m spacy download en_core_web_sm

After updating **textnets** you may also need to update the language models.
Run the following command to check::

   $ python -m spacy validate

.. _`language models`: https://spacy.io/usage/models#download

If there are no language models available for your corpus language, there may
be some `basic support <https://spacy.io/usage/models#languages>`_. Even in
that case, some languages (including Korean, Vietnamese, Thai, Russian, and
Ukrainian) require additional installs for tokenization support. Consult the
spaCy documentation for details.

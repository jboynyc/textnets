.. highlight:: shell

============
Installation
============

**textnets** is included in the `Python Package Index`_ and `nixpkgs`_. That
means you can install the package using `pip`_ or `nix`_.

.. _`Python Package Index`: https://pypi.org/project/textnets/
.. _`nixpkgs`: https://search.nixos.org/packages?query=textnets
.. _pip: https://pip.pypa.io
.. _nix: https://nixos.org

.. note::

   Please note that **textnets** requires Python 3.10 or newer to run.

Using pip
---------

Alternately, if you already have Python installed, you can use its package
manger to install **textnets**. In a `virtual environment`_, run::

   $ python -m pip install textnets

.. _`virtual environment`: https://packaging.python.org/tutorials/installing-packages/#creating-virtual-environments

Using nix
---------

Users of ``nix`` can use the version from nixpkgs, for instance by using
``nix-shell``::

   $ nix-shell -p 'python3.withPackages (p: with p; [ ipython textnets spacy_models.en_core_web_sm ])' --run ipython

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
that case, some languages (including Japanese, Russian, Thai, Vietnamese,
Ukrainian, and Chinese) require additional installs for tokenization support.
Consult the spaCy documentation for details.

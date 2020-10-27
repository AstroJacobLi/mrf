Installation
============
``mrf`` is a Python package which also incorperates some ``iraf`` functions. You can download and install the package as follows. You must have `galsim <https://github.com/GalSim-developers/GalSim>`_ installed in advance. You will also need `unagi <https://github.com/dr-guangtou/unagi>`_ to download HSC images.

Install from source code (Recommended)
--------------------------
The source code of ``mrf`` is stored in GitHub repo https://github.com/AstroJacobLi/mrf. If you have configured ``git`` in your computer, you can make a new directory for this package and then clone the GitHub repository locally.

.. code-block:: bash

  $ mkdir <install dir>
  $ cd <install dir>
  $ git clone git@github.com:AstroJacobLi/mrf.git
  $ cd mrf
  $ python setup.py install

If you don't have ``git`` configured, you can also download the ``zip`` file directly from https://github.com/AstroJacobLi/mrf/archive/master.zip, then unzip it and install it in the same way using ``$ python setup.py install``. 


Install from ``pip``
----------------------
Probably the easiest way is to install with ``pip``. The version you installed from ``pip`` is stable, but may lack some new functions. If you find some bugs, please try to install the most up-to-date ``mrf`` from source code, or open an issue on GitHub.

.. code-block:: bash

  $ pip install -U mrf

.. warning::
   We are sorry that MRF has not been fully compatible with Windows. Please use Linux or MacOS!

Test the Installation
-----------------------
The following snippet checks the availability of some ``iraf`` files. ``True`` means you have installed ``mrf`` successfully! Bravo!

.. code-block:: python

    import mrf, os
    print(os.path.isfile(os.path.join(mrf.__path__[0], 'iraf/macosx/x_images.e')))

Requirements
-------------
``Python>=3`` is needed, but you can try whether ``mrf`` still works under ``python2``. Check out other dependences of ``mrf`` from `requirements.txt <https://github.com/AstroJacobLi/mrf/blob/master/requirements.txt>`_.
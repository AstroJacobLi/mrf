Installation
============
``mrf`` is a Python package which also incorperates some ``iraf`` functions. You can download and install the package as follows.

Install from source code
------------------------
The source code of ``mrf`` is stored in GitHub repo https://github.com/AstroJacobLi/mrf. If you have configured ``git`` in your computer, you can make a new directory for this package and then clone the GitHub repository locally.

.. code-block:: bash

  $ mkdir <install dir>
  $ cd <install dir>
  $ git clone git@github.com:AstroJacobLi/mrf.git
  $ cd mrf
  $ python setup.py install

If you don't have ``git`` configured, you can also download the ``zip`` file directly from https://github.com/AstroJacobLi/mrf/archive/master.zip, then unzip it and install it in the same way using ``$ python setup.py install``. 


Install from ``pip``
--------------------
To be done in the future.


Test the Installation
---------------------
**You need to export the path of** ``mrf`` **as an environment variable**. 

So please open ``~/.bash_profile`` (or ``~/.bashrc``) with your favourite editor, and write 

.. code-block:: bash

  $ export PYTHONPATH=$PYTHONPATH:"<install dir>"
  
to it (replace ``<install dir>`` with your certain directory). Don't forget to validate it by 

.. code-block:: bash

  $ source ~/.bash_profile


Then import ``mrf`` package in Python. The following snippet check the availability of some ``iraf`` files. The output should be ``"<install dir>/mrf/mrf/iraf/macosx/"``.

.. code-block:: python

    import mrf
    print(mrf.__file__.rstrip('__init__.py') + 'iraf/macosx/') 
    # It should be "<install dir>/mrf/mrf/iraf/macosx/"
    # otherwise the environmental variable is not set correctly.

Requirements
-------------
``Python>=3`` is needed, but you can try whether ``mrf`` still works under ``python2``. Check out the dependence of ``mrf`` depends from `requirements.txt <https://github.com/AstroJacobLi/mrf/blob/master/requirements.txt>`_.
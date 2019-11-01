MRF: Multi-Resolution Filtering
===============================
Multi-Resolution Filtering is a method for isolating faint, extended emission in `Dragonfly <http://dragonflytelescope.org>`_ data and other low resolution images. It is implemented in an open-source MIT licensed Python package ``mrf``. Please read `van Dokkum et al. (2019) <https://arxiv.org/abs/1910.12867>`_ for the methodology and description of implementation.

.. image:: https://img.shields.io/badge/license-MIT-blue
    :target: https://opensource.org/licenses/mit-license.php
    :alt: License

.. image:: https://readthedocs.org/projects/mrfiltering/badge/?version=latest
    :target: https://mrfiltering.readthedocs.io/en/latest/?badge=latest
    :alt: Documentation Status

.. image:: https://img.shields.io/badge/version-1.0.4-green
    :alt: Version

.. image:: https://img.shields.io/badge/arXiv-1910.12867-blue
    :target: https://arxiv.org/abs/1910.12867
    :alt: arXiv

.. image:: https://img.shields.io/badge/GitHub-astrojacobli%2Fmrf-blue
    :target: https://github.com/AstroJacobLi/mrf
    :alt: GitHub Repo

.. image:: https://img.shields.io/github/repo-size/astrojacobli/mrf
    :target: https://github.com/AstroJacobLi/mrf
    :alt: Repo Size


Basic Usage
-----------
.. code-block:: python

    from mrf.task import MrfTask
    task = MrfTask('m101-df3-task.yaml')
    img_lowres = 'M101_DF3_df_r.fits'
    img_hires_b = 'M101_DF3_cfht_r.fits'
    img_hires_r = 'M101_DF3_cfht_r.fits'
    certain_gal_cat = 'gal_cat_m101.txt'
    results = task.run(img_lowres, img_hires_b, img_hires_r, certain_gal_cat, 
                    output_name='m101_df3', verbose=True)
    results.lowres_final.display_image()

.. figure:: https://github.com/AstroJacobLi/mrf/raw/master/examples/M101-DF3/m101-df3-demo.png
    :width: 640px
    :align: center
    :alt: alternate text
    :figclass: align-center
    
Please check :ref:`Tutorials` for more details.

User Guide
-----------

.. toctree::
   :maxdepth: 2

   guide/install
   tutorial/mrf-tutorial
   

.. toctree::
   :maxdepth: 1

   tutorial/configuration
   tutorial/misc
   license
   guide/changelog


Index
------------------

* :ref:`modindex`
* :ref:`search`


Citation
--------
``mrf`` is a free software made available under the MIT License by `Pieter van Dokkum <http://pietervandokkum.com>`_ (initial development) and `Jiaxuan Li <https://astrojacobli.github.io>`_ (implementation, maintenance, and documentation). If you use this package in your work, please cite `van Dokkum et al. (2019) <https://arxiv.org/abs/1910.12867>`_.

You are welcome to report bugs in ``mrf`` via creating issues at https://github.com/AstroJacobLi/mrf/issues.

Need more help? Feel free to contact via pieter.vandokkum@yale.edu and jiaxuan_li@pku.edu.cn.


Acknowledgment
---------------
Many scripts and snippets are from `kungpao <https://github.com/dr-guangtou/kungpao>`_ (mainly written by `Song Huang <http://dr-guangtou.github.io>`_). `Johnny Greco <http://johnnygreco.github.io>`_ kindly shared his idea of the code structure. `Roberto Abraham <http://www.astro.utoronto.ca/~abraham/Web/Welcome.html>`_ found the first few bugs of this package and provided useful solutions. Here we appreciate their help!
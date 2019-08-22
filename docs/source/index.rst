MRF: Multi-Resolution Filtering
===============================
Multi-Resolution Filtering is a method for isolating faint, extended emission in `Dragonfly <http://dragonflytelescope.org>`_ data and other low resolution images. It is implemented in an MIT licensed Python package ``mrf``. 


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

.. figure:: https://github.com/AstroJacobLi/mrf/raw/master/examples/m101-df3-final.png
    :width: 700px
    :align: center
    :alt: alternate text
    :figclass: align-center

Guide
-----

.. toctree::
   :maxdepth: 2

   guide/install
   tutorial/mrf-tutorial

   license

Citation
--------

Please cite van Dokkum et al. (in prep). Need more help? Feel free to contact via pieter.vandokkum@yale.edu and jiaxuan_li@pku.edu.cn.


Indices and tables
------------------

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`

Examples
--------

This example shows the tidal feature of NGC 5907, described in `van Dokkum et al. (2019) <https://ui.adsabs.harvard.edu/abs/2019arXiv190611260V/abstract>`_ . The images presented there just used this algorithm. Full resolution Dragonfly images and MRF results can be found `here <https://www.pietervandokkum.com/ngc5907>`_. Check `this notebook <https://github.com/AstroJacobLi/mrf/blob/master/examples/mrfTask-n5907.ipynb>`_ for more details in how to do MRF using this Python package!

.. figure:: https://github.com/AstroJacobLi/mrf/raw/master/examples/n5907-demo.png
    :width: 1000px
    :align: center
    :alt: alternate text
    :figclass: align-center


The next example shows how powerful MRF is in extracting low surface brightness features. The ultra-diffuse galaxy M101-DF3 is revealed by MRF after subtracting compact objects and bright star halos according to `van Dokkum et al. (in prep) <https://www.pietervandokkum.com>`_ . Check `this notebook <https://github.com/AstroJacobLi/mrf/blob/master/examples/mrfTask-m101df3.ipynb>`_ for more details.

.. figure:: https://github.com/AstroJacobLi/mrf/raw/master/examples/m101-df3-demo.png
    :width: 1000px
    :align: center
    :alt: alternate text
    :figclass: align-center


You can also use `this script <https://github.com/AstroJacobLi/mrf/blob/master/examples/mrf-task.py>`_ to run the MRF task. Take NGC 5907 as an example:

.. code-block:: python

    python mrf-task.py n5907_df_g.fits ngc5907_cfht_g.fits ngc5907_cfht_r.fits ngc5907-task.yaml --galcat='gal_cat_n5907.txt' --output='n5907_g'
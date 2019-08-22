Tutorials
---------
This page shows the applications of MRF on various objects. You can download the data and implement MRF by yourself. 

NGC 5907
^^^^^^^^^
NGC 5907 is a edge-on spiral galaxy, which is famous for its prominent tidal streams. Recently the images taken by Dragonfly Telephoto Array reveal more details about the low surface brightness features around this galaxy (`van Dokkum et al. 2019 <https://ui.adsabs.harvard.edu/abs/2019arXiv190611260V/abstract>`_). In this paper, we isolate diffuse low surface brightness streams using "Muilt-Resolution Filtering" (MRF). Now we show how to do it using Python package ``mrf``. Check `this notebook <https://github.com/AstroJacobLi/mrf/blob/master/examples/mrfTask-n5907.ipynb>`_ for the whole process of MRF using this Python package. Below I briefly show the key steps.

Full resolution Dragonfly images and MRF results can be found `here <https://www.pietervandokkum.com/ngc5907>`_. The corresponding CFHT images can be found on `Canadian Astronomy Data Center <http://www.cadc-ccda.hia-iha.nrc-cnrc.gc.ca/en/search/?collection=CFHTMEGAPIPE&noexec=true#queryFormTab>`_. Besides, ``mrf`` provides functions to download high-resolution images (including `CFHT <https://www.cfht.hawaii.edu>`_ and `HSC <https://hsc.mtk.nao.ac.jp>`_), shown as follows:

.. code-block:: python

    # Position of NGC 5907
    ra, dec = 228.974042, 56.328771

    # Notice: the images might be very large (several Gb)!
    from mrf import download
    download.download_highres('n5907_df_g.fits', high_res='cfht', band='g')
    download.download_highres('n5907_df_r.fits', high_res='cfht', band='r')
    # CFHT images will be saved as "CFHT_megapipe_img_G.fits" 
    # and "CFHT_megapipe_img_R.fits".

After downloading high-resolution images, you need to bin it by a :math:`2\times2` pixel box and convolve with a :math:`\sigma=1` pixel Gaussian kernel to mitigate projection errors. 

.. code-block:: python

    from mrf.celestial import Celestial
    from astropy.convolution import convolve, Gaussian2DKernel
    hdu = fits.open('CFHT_megapipe_img_G.fits')
    cfht = Celestial(hdu[0].data, header=hdu[0].header)
    hdu.close()
    cfht.resize_image(0.5) # bin with 2*2 box
    cfht.image = convolve(cfht.image, Gaussian2DKernel(1))
    cfht.save_to_fits('ngc5907_cfht_g.fits')
    # Then do the same for R-band image

The main part of MRF can be simply done by the following code. A configuration YAML file is needed to provide parameters for relevant functions such as SExtractor (``sep``). Check `here <https://github.com/AstroJacobLi/mrf/blob/master/examples/ngc5907-task.yaml>`_ for more explanation on configuration file. If you want to retain certain galaxies during MRF, make an ASCII catalog which contains the ``RA`` and ``DEC`` of galaxies (see `gal_cat_n5907.txt <https://github.com/AstroJacobLi/mrf/blob/master/examples/gal_cat_n5907.txt>`_ for an example). If you don't want to reserve any galaxy, just leave ``certain_gal_cat = None``.

.. code-block:: python

    from mrf.task import MrfTask
    task = MrfTask('ngc5907-task.yaml')
    img_lowres = 'n5907_df_g.fits'
    img_hires_b = 'ngc5907_cfht_g.fits'
    img_hires_r = 'ngc5907_cfht_r.fits'
    certain_gal_cat = 'gal_cat_n5907.txt'
    results = task.run(
        img_lowres, img_hires_b, img_hires_r, certain_gal_cat, 
        output_name='n5907', verbose=True)

``results`` has many attributes including the MRF results ``results.lowres_final``, the models ``results.lowres_model``, the convolution kernel ``results.kernel_med``, the stacked PSF ``results.PSF``, etc.

.. code-block:: python

    import matplotlib.pyplot as plt
    fig, [ax1, ax2, ax3] = plt.subplots(1, 3, figsize=(15, 8))

    ax1 = display_single(results.lowres_input.image, ax=ax1, 
        pixel_scale=2.5, scale_bar_length=300, scale_bar_y_offset=0.3,
        add_text='NGC\, 5907', text_y_offset=0.65)

    ax2 = display_single(results.lowres_model.image, ax=ax2, 
        scale_bar=False, add_text='Model', text_y_offset=0.65)

    ax3 = display_single(results.lowres_final.image, ax=ax3, 
        scale_bar=False, add_text='Redisual', text_y_offset=0.65)

    plt.subplots_adjust(wspace=0.05)
    plt.savefig('n5907-demo.png', bbox_inches='tight', facecolor='silver')
    plt.show()

.. figure:: https://github.com/AstroJacobLi/mrf/raw/master/examples/n5907-demo.png
    :width: 1000px
    :align: center
    :alt: alternate text
    :figclass: align-center

M101-DF3
^^^^^^^^^
M101-DF3 is a satellite galaxy of the famous spiral galaxy M101, presented in `Merritt, van Dokkum, & Abraham 2014 <https://iopscience.iop.org/article/10.1088/2041-8205/787/2/L37/meta>`_. It has an effective surface brightness :math:`\mu_g=27.4\pm0.2` and effective radius :math:`r_e=30\pm 3` arcsec. 

In this example, the dwarf galaxy M101-DF3 is revealed clearly by MRF after subtracting compact objects and bright star halos according to `van Dokkum et al. (in prep) <https://www.pietervandokkum.com>`_ . The basic procedures are the same as NGC 5907 example. Check `this notebook <https://github.com/AstroJacobLi/mrf/blob/master/examples/mrfTask-m101df3.ipynb>`_ for more details. 

.. figure:: https://github.com/AstroJacobLi/mrf/raw/master/examples/m101-df3-demo.png
    :width: 1000px
    :align: center
    :alt: alternate text
    :figclass: align-center

After having ``results``, you can show the flux model, kernels and PSF as follows. 

.. code-block:: python

    fig, [ax1, ax2] = plt.subplots(1, 2, figsize=(15, 8))
    ax1 = display_single(results.lowres_input.image, ax=ax1, 
                         scale_bar=False, add_text='Kernel')
    ax2 = display_single(results.hires_fluxmod, ax=ax2, 
                         scale='percentile', lower_percentile=0.5,
                         scale_bar=False, add_text='FLux\,Model')
    plt.savefig('m101-df3-fluxmodel.png', bbox_inches='tight')
    plt.show()
    
.. figure:: https://github.com/AstroJacobLi/mrf/raw/master/examples/m101-df3-fluxmodel.png
    :width: 1000px
    :align: center
    :alt: alternate text
    :figclass: align-center

.. code-block:: python

    fig, [ax1, ax2] = plt.subplots(1, 2, figsize=(15, 8))
    ax1 = display_single(results.kernel_med, ax=ax1, 
                        scale_bar=False, add_text='Kernel')
    ax2 = display_single(results.PSF, ax=ax2, 
                        scale_bar=False, add_text='PSF')
    plt.savefig('m101-df3-kernel-psf.png', bbox_inches='tight')
    plt.show()

.. figure:: https://github.com/AstroJacobLi/mrf/raw/master/examples/m101-df3-kernel-psf.png
    :width: 1000px
    :align: center
    :alt: alternate text
    :figclass: align-center


You can also use `this script <https://github.com/AstroJacobLi/mrf/blob/master/examples/mrf-task.py>`_ to run the MRF task. Take NGC 5907 as an example:

.. code-block:: python

    python mrf-task.py n5907_df_g.fits ngc5907_cfht_g.fits ngc5907_cfht_r.fits ngc5907-task.yaml --galcat='gal_cat_n5907.txt' --output='n5907_g'
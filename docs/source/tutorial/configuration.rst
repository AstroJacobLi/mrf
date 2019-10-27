Configuration file
-------------------
A configuration file is needed to run MRF properly. In this page we guide you through the configuration file and explain some important parameters.

The configuration file has several sections. Under each section several important parameters are listed after a colon. Please see `here <https://github.com/AstroJacobLi/mrf/blob/master/examples/NGC5907/ngc5907-task.yaml>`_ as an example. 

TL;DR
^^^^^^
* Please write pixel scales, zeropoints correctly. 
* ``frac_maxflux`` is very important, you need to adjust this parameter several times to make the residual image cleanest.
* ``fluxmodel.unmask_lowsb``, ``fluxmodel.sb_lim`` and ``fluxmodel.unmask_ratio`` are important for discovering low-SB objects! Please follow the instructions in the corresponding section below.
* If the stars are dirty, try to adjust ``starhalo.n_stack`` to a smaller number, or make your field larger to contain more bright stars. 
* You can iterate mask size using ``mrf.utils.adjust_mask``.

``hires`` and ``lowres``
^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: yaml

    hires:
        dataset: 'cfht'
        zeropoint: 30.0
        pixel_scale: 0.372

    lowres:
        dataset: 'df'
        band: 'r'
        pixel_scale: 2.5
        sub_bkgval: True # highly recommend True for Dragonfly
        magnify_factor: 3.0
        zeropoint: 27.0 
        color_term: 0.0

These two sections are related to basic information of low-resolution and high-resolution datasets you use. You need to fill in the photometry zeropoint, pixel scale (in the unit of arcsec/pixel) of each dataset. The original pixel scales of several widely-used high-resolution surveys are listed below (in the unit of arcsec/pixel). 

=======  ============
survey   pixel scale
=======  ============
SDSS      0.395
DECaLS    0.262
CFHT      0.186
HSC       0.168
=======  ============

However, we recommend the user to bin the original high-resolution images with a 2*2 box and convolve with a 1 pixel Gaussian kernel before running MRF. Thus the pixel scale of input high-resolution image will be double the original pixel scale. **Incorrect pixel scale can cause completely wrong results. Be aware of the pixel scale of the image you are processing, and remember to pass the pixel scale to the configuration file in time.**

If you use Dragonfly as low-resolution image, we recommend to subtract a global sky background by setting ``sub_bkgval: True`` (which will search keyword ``BACKVAL`` in the header of input image). If you use other datasets, you need to subtract sky manually before MRF and set ``sub_bkgval: False``. 

``magnify_factor: 3.0`` indicates the factor of magnification on low-resolution image. Typically 3.0 works well. ``color_term`` corresponds to the filter-correction term (see Equation 5 in Section 3.1 of van Dokkum et al. in prep). We derive this term based on both synthetic stellar photometry and empirical photometry on images (analyses on more surveys are underway). 

+----------+----------+--------------------+--------------------+
| hires    | lowres   | r-band             | g-band             |
+==========+==========+====================+====================+
| DECaLS   |   DF     | | 0.10 (empirical) | | 0.06 (empirical) |
|          |          | | 0.07 (synthetic) | | 0.03 (synthetic) |
+----------+----------+--------------------+--------------------+
| CFHT     |   DF     | | 0.01 (empirical) | | 0.05 (empirical) |
|          |          | | 0.00 (synthetic) | | 0.05 (synthetic) |
+----------+----------+--------------------+--------------------+

``sex``, ``fluxmodel`` and ``kernel``
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: yaml

    sex:
        sigma: 2.5
        minarea: 2
        b: 64
        f: 3
        deblend_cont: 0.005
        deblend_nthresh: 32
        sky_subtract: True
        flux_aper: [3, 6] # pixel
        show_fig: False

    fluxmodel:
        gaussian_radius: 1.5     # conv = convolve(mask, Gaussian2DKernel(1.5))
        gaussian_threshold: 0.05 # mask = conv > 0.05
        unmask_lowsb: False
        sb_lim: 26.0
        unmask_ratio: 2.0
        interp: 'cubic'

    kernel:
        kernel_size: 8 # In original coordinate, before magnification
        kernel_edge: 1
        nkernel: 25
        frac_maxflux: 0.1
        circularize: False
        show_fig: True
        minarea: 25

``sex`` is the abbreviation of `Source Extractor <https://www.astromatic.net/software/sextractor>`_, which is widely used in object detections. In MRF we use a Python-version of Source Extractor: `sep <http://sep.readthedocs.io>`_. Parameters under ``sex`` are related to source extraction. We refer the user to the `SExtractor Manual <https://www.astromatic.net/pubsvn/software/sextractor/trunk/doc/sextractor.pdf>`_ for detailed meaning of ``sigma, minarea, b, f, deblend_cont, deblend_nthresh``. 

``sky_subtract: True`` means ``sep`` subtracts a locally measured 2-D map of sky from the image, then identifies objects from the residual image. Thus ``b`` is crucial for removing compact objects from low-SB objects. Fine mesh (which is used to estimate local sky map) will subtract smooth components of an object, leaving compact objects to be detected. If you only want to extract very compact objects, small ``b`` will be helpful. Otherwise you should use large ``b`` to avoid subtraction of extended galaxies you want. ``sep`` is able to measure flux within an annulus, ``flux_aper`` indicates the (inner and outer) radii of annulus in the unit of pixel. We use flux within [3 pix, 6 pix] to normalize stars for stacking PSF. 

Section ``fluxmodel`` and ``kernel`` controls the key process in MRF, please see Section 3.2 - Section 3.5 of van Dokkum et al. in prep for details). First we build a mask based on the segmentation map from ``sep``. Then we enlarge this mask by convolving a Gaussian kernel with ``gaussian_radius: 1.5`` pixels and mask out pixels whose value are below ``gaussian_threshold: 0.05``. We don't recommend changing this two parameters. 

``unmask_lowsb`` is crucial for identifying low-SB extended emissions. There are two cases that you want to use MRF. First, you need to remove compact objects and stars from a given object. Second, you need to discover new low-SB extended objects in a given image. In the latter case, you may need to turn on ``unmask_lowsb``. This removes objects below certain surface brightness threshold (``sb_lim: 26.0``, in the unit of mag/arcsec^2) and objects extended enough (``unmask_ratio: 3``). In van Dokkum et al. (in prep), we define "degree of spacial extent" by 

.. math::

   E =  \frac{F^{\text{H}(3)}}{F^{\text{H}(3)} * K},

where :math:`F^{\text{H}(3)}` is the flux model, and :math:`K` is the kernel. If :math:`\langle E \rangle \ll 1`, it is a compact object that should be retained in the flux model and subtracted from the Dragonfly data. Hence we retain (compact) objects in flux model by :math:`E > \texttt{unmask_ratio}`. Small ``unmask_ratio`` leaves very extended objects in the final product. The value of ``unmask_ratio`` and ``sb_lim`` depends on your science goals. We don't want to retain some small and compact objects. ``minarea`` is the minimum area (in Dragonfly pixels) of objects that are retained.

Interpolation of images is important in MRF. We provide several interpolation methods including ``'iraf', 'cubic', 'lanczos', 'quintic'``. IRAF method uses 3rd order polynomial interpolation, which might not work under Windows system. However we recommend using IRAF interpolation under most circumstances. When using the other three, you may see crosses around very bright stars. 

Parameters in ``kernel`` section are very important. ``kernel_size`` is the size of kernel in the original low-resolution image coordinate (before magnification). For example, ``kernel_size: 8`` and ``magnify_factor: 3.0`` means the actual kernel is 24 pixel * 24 pixel. ``nkernel: 25`` indicates that the kernel will be generated based on 25 objects. Only objects fainter than "``frac_maxflux`` * flux of fifth-brightest object" will be used. Hence ``frac_maxflux`` is very important, you need to adjust this parameter several times to make the residual image cleanest. Typically it should be less than 0.3. Please note that it could be different between bands. The kernel will be circularized if ``circularize: True``, however it's not necessary to circularize kernel in most cases. 

``starhalo``
^^^^^^^^^^^^^

.. code-block:: yaml

    starhalo:
        bright_lim: 17.5 # only stack stars brighter than bright_lim
        fwhm_lim: 50 # only stack stars whose FWHM < fwhm_lim
        n_stack: 10
        halosize: 30 # radial size, in pixel, on low-res image. Star cutout size will be 2 * halosize + 1
        padsize: 50
        edgesize: 3
        norm: 'flux_ann' # or 'flux' or 'flux_auto'
        b: 32
        f: 3
        sigma: 4
        minarea: 5
        deblend_cont: 0.005
        deblend_nthresh: 32
        sky_subtract: True
        flux_aper: [3, 6] # pixels
        mask_contam: True
        cval: 'nan'
        interp: 'iraf'

Parameters in this section are used to stack PSF using bright stars. The PSF will further be used to subtract bright stars from the image. We already identified bright stars on low-resolution image using ``sep``, and here we only select stars brighter than ``bright_lim`` and FWHM less than ``fwhm_lim``, avoiding too saturated stars. The maximum number of stars selected is ``n_stack`` (typically 10-20, it's not good to use very large number of stars). We make a cutout of each star with a ``2 * halosize + 1`` pixel width square. Since stars have different brightness, we normalize each cutout using either the total flux measured by ``sep`` (i.e. ``norm: 'flux'``) or the flux within a certain annulus (i.e. ``norm: 'flux_ann'``). The default annulus is between 3 pix and 6 pix, since the saturation peak (if exists) drops quickly before 3 pixels. You can adjust the annulus size in ``flux_aper``. 

After making a cutout of a star, you may need to mask out contaminations around it by indicating ``mask_contam: True``. If so, the masked region will be filled with ``cval``, which could be any float number or `nan`. The ``interp`` parameter means the same as in ``fluxmodel`` section.


``clean``
^^^^^^^^^^^^^

.. code-block:: yaml

    clean:
        clean_img: True
        clean_file: False
        replace_with_noise: False
        gaussian_radius: 1.5
        gaussian_threshold: 0.001
        bright_lim: 17.5
        r: 5.0

Now we have already subtracted both compact objects and bright stars in the field. To make things neat, we apply masks on the residual image by indicating ``clean_img: True``. We generate mask by convolving the segmentation map with a ``gaussian_radius: 1.5`` Gaussian kernel and filtering it with a threshold ``gaussian_threshold: 0.001``. This threshold is typically around 0.001. Larger radius and smaller threshold give you more aggressive mask. We additionally mask out bright stars (brighter than ``bright_lim: 17.5``) by drawing an ellipse on the image with a blow-up factor ``r: 5.0``. You can adjust the mask afterward using `mrf.utils.adjust_mask <https://mrfiltering.readthedocs.io/en/latest/api.html#mrf.utils.adjust_mask>`_ function.

Since MRF creates many temporary files whose names star with an underline (such as ``_median_psf.fits``), we remove these files by indicating ``clean_file: True``. 
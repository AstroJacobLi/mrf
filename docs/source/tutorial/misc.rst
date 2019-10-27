Miscellaneous Functions
------------------------

``mrf.sbcontrast.cal_sbcontrast``
++++++++++++++++++++++++++++++++++
This function calculates the surface brightness detection limit on a given angular scale. It requires the image and corresponding mask which masks out contamination after MRF. The usage is as follows:

.. code-block:: python

    _ = cal_sbcontrast(imaeg, mask, zeropoint=27.31, pixel_scale=2.5, 
                       scale_arcsec=60, minfrac=0.8, minback=6)

Then it prints out the SB detection limit on the scale of 60 arcsec * 60 arcsec. You can also pass ``scale_arcsec=[10, 30, 60]`` to get SB limit on these three scales. 


``mrf.celestial.Celestial``
+++++++++++++++++++++++++++++
This class is a convenient tool for you to manipulate astronomical images. It is almost a celestial body from an observational perspective. It has its image, header, WCS. The mask which masks out contaminations can also be stored as an attribute. Then this ``Celestial`` object can be saved to FITS file, can be shifted, resized, rotated, etc. What's more, the user could check the image/mask/masked image simply by invoke ``Celestial.display_image()``.

``mrf.celestial.Star``
++++++++++++++++++++++++
This ``Star`` class is the inheritance of ``Celestial`` class. It represents a small cutout, which is typically a star. Other than the functions inherited from ``Celestial``, ``Star`` object has extra functions such as ``centralize``, ``mask_out_contam``. One can use this class to do PSF stacking easily. 

``mrf.utils.phys_size`` 
+++++++++++++++++++++++++
This function gives you a nice API to calculate the physical scale at given redshift. 

``mrf.utils.azimuthal_average``
++++++++++++++++++++++++++++++++
You can get azimuthal average profile of a 2-D array by using this function.


``mrf.utils.extract_obj``
++++++++++++++++++++++++++
This function builds on ``sep.extract``, and has new features such as measure AUTO magnitude (as in SExtractor) and FWHM of an object. It returns an objects catalog along with a segmentation map.


``mrf.display.display_single``
++++++++++++++++++++++++++++++++
This function gives you the chance to display astronomical figures correctly in a cell of Jupyter Notebook. It uses ``zscale`` and ``arcsinh`` stretch. I like this function a lot.

Changelog
---------
* 2019-09-07: Version 1.0.2 released. Now users don't need to configure environmental path themselves. External IRAF files are automatically installed. This version also removes strong dependence on IRAF, while users could still using IRAF methods if they have many saturated stars and large images. Sometimes it would be wrong when using IRAF method to magnify image then using ``reproject``. If the high-res image after color correction is all NaN, please use other interpolation methods such as `lanczos` or `cubic`. We also add filter correction dictionary in ``__init__.py``.

* 2019-08-23: Version 1.0.1 released.


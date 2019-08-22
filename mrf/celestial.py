import os
import copy
import scipy

import numpy as np
import matplotlib.pyplot as plt

from astropy import wcs
from astropy.io import fits
from astropy.table import Table, Column
import astropy.units as u
from astropy.coordinates import SkyCoord
from .display import display_single, SEG_CMAP
from .utils import img_cutout
from .imtools import imshift, imdelete, magnify, blkavg

class Celestial(object):
    '''
    Class for ``Celestial`` object. 
    This class is basically a celestial body from observational perspective. 
    It has its image, header, WCS. The mask which masks out contaminations can also be stored as an attribute. 
    Then this ``Celestial`` object can be saved to FITS file, can be shifted, resized, rotated, etc. 
    What's more, the user could check the image/mask/masked image simply by invoke ``Celestial.display_image()``.
    
    This class can also be inherited to form other classes.
    '''

    def __init__(self, img, mask=None, header=None, dataset='Dragonfly'):
        '''
        Initialize ``Celestial`` object.

        Parameters:
            img (numpy 2-D array): image array.
            mask (numpy 2-D array, optional): mask array. 1 means the pixel will be masked.
            header: header of image, containing WCS information. Typically it is ``astropy.io.fits.header`` object.
            dataset (str): The description of the input data.

        Returns:
            None
        '''
        try:
            self.pixel_scale = abs(header['CD1_1'] * 3600)
        except:
            self.pixel_scale = abs(header['PC1_1'] * 3600)
        self.shape = img.shape # in ndarray format
        self.dataset = dataset
        self._image = img
        if mask is not None:
            self._mask = mask
        # Sky position
        ny, nx = img.shape
        self.ny = ny
        self.nx = nx
        if header is not None:
            self.header = header
            self.wcs = wcs.WCS(header)
            self.ra_cen, self.dec_cen = list(map(float, self.wcs.wcs_pix2world(ny // 2, nx // 2, 0)))
            # This follows lower-left, lower-right, upper-right, upper-left.
            self.ra_bounds, self.dec_bounds = self.wcs.wcs_pix2world([0, img.shape[1], img.shape[1], 0], 
                                                [0, 0, img.shape[0], img.shape[0]], 0)
            self.sky_bounds = np.append(self.ra_bounds[2:], self.dec_bounds[1:3])
        self.scale_bar_length = 5 # initial length for scale bar when displaying

    @property
    def image(self):
        return self._image
    
    @image.setter
    def image(self, img_array):
        self._image = img_array

    @property
    def mask(self):
        return self._mask
    
    @mask.setter
    def mask(self, mask_array):
        self._mask = mask_array

    @property
    def hscmask(self):
        return self._mask
    
    @hscmask.setter
    def hscmask(self, mask_array):
        self._hscmask = mask_array

    @property
    def variance(self):
        return self._variance
    
    @variance.setter
    def variance(self, variance_array):
        self._variance = variance_array

    # Save 2-D numpy array to ``fits``
    def save_to_fits(self, fits_file_name, data='image', overwrite=True):
        """
        Save image or mask of this ``Celestial`` object to ``fits`` file.

        Parameters:
            fits_file_name (str): File name of ``fits`` file
            data (str): can be 'image' or 'mask'
            overwrite (bool): Default is True

        Returns:
            None
        """
        if data == 'image':
            data_use = self.image
        elif data == 'mask':
            data_use = self.mask
        else:
            raise ValueError('Data can only be "image" or "mask".')
        img_hdu = fits.PrimaryHDU(data_use)

        if self.header is not None:
            img_hdu.header = self.header
            if self.wcs is not None:
                wcs_header = self.wcs.to_header()
                import fnmatch
                for i in wcs_header:
                    if i in self.header:
                        self.header[i] = wcs_header[i]
                    if fnmatch.fnmatch(i, 'PC?_?'):
                        self.header['CD' + i.lstrip("PC")] = wcs_header[i]
                img_hdu.header = self.header
        elif self.wcs is not None:
            wcs_header = self.wcs.to_header()
            img_hdu.header = wcs_header
        else:
            img_hdu = fits.PrimaryHDU(data_use)

        if os.path.islink(fits_file_name):
            os.unlink(fits_file_name)

        img_hdu.writeto(fits_file_name, overwrite=overwrite)
        return img_hdu
    
    # Shift image/mask
    def shift_image(self, dx, dy, method='iraf', order=5, cval=0.0):
        '''Shift the image of Celestial object. The WCS of image will also be changed.

        Parameters:
            dx (float): shift distance (in pixel) along x (horizontal). 
                Note that elements in one row has the same y but different x. 
                Example: dx = 2 is to shift the image "RIGHT" (as seen in DS9), dy = 3 is to shift the image "UP".
            dy (float): shift distance (in pixel) along y (vertical). 
                Note that elements in one row has the same y but different x. 
                Example: dx = 2 is to shift the image "RIGHT" (as seen in DS9), dy = 3 is to shift the image "UP".
            method (str): interpolation method. Use 'lanczos' or 'iraf'. 
                If using 'iraf', default interpolation is 'poly3. 'Lanczos' requires ``GalSim`` installed.
            order (int): the order of Lanczos interpolation (>0).
            cval (float): value to fill the edges. Default is 0.

        Returns:
            shift_image (ndarray): shifted image, the "image" attribute of ``Celestial`` class will also be changed accordingly.
        '''

        ny, nx = self.image.shape
        if abs(dx) > nx or abs(ny) > ny:
            raise ValueError('# Shift distance is beyond the image size.')
        if method == 'lanczos':
            try: # try to import galsim
                from galsim import degrees, Angle
                from galsim.interpolant import Lanczos
                from galsim import Image, InterpolatedImage
                from galsim.fitswcs import AstropyWCS
            except:
                raise ImportError('# Import ``galsim`` failed! Please check if ``galsim`` is installed!')
            # Begin shift
            assert (order > 0) and isinstance(order, int), 'order of ' + method + ' must be positive interger.'
            galimg = InterpolatedImage(Image(self.image, dtype=float), 
                                    scale=self.pixel_scale, x_interpolant=Lanczos(order))
            galimg = galimg.shift(dx=dx * self.pixel_scale, dy=dy * self.pixel_scale)
            result = galimg.drawImage(scale=self.pixel_scale, nx=nx, ny=ny)#, wcs=AstropyWCS(self.wcs))
            self._image = result.array
            # Change the WCS of image
            hdr = copy.deepcopy(self.header)
            hdr['CRPIX1'] += dx
            hdr['CRPIX2'] += dy
            self.header = hdr
            self.wcs = wcs.WCS(hdr)
            self._wcs_header_merge()
            return result.array
        elif method == 'iraf':
            self.save_to_fits('./_temp.fits', 'image')
            imshift('./_temp.fits', './_shift_temp.fits', dx, dy, interp_type='poly3', boundary_type='constant')
            hdu = fits.open('./_shift_temp.fits')
            self.image = hdu[0].data
            self.shape = hdu[0].data.shape
            self.header = hdu[0].header
            self.wcs = wcs.WCS(self.header)
            hdu.close()
            imdelete('./*temp.fits')
        else:
            raise ValueError("# Not supported interpolation method. Use 'lanczos' or 'iraf'.")

    def shift_mask(self, dx, dy, method='iraf', order=5, cval=0.0):
        '''Shift the mask of Celestial object.

        Parameters:
            dx (float): shift distance (in pixel) along x (horizontal). 
                Note that elements in one row has the same y but different x. 
                Example: dx = 2 is to shift the mask "RIGHT" (as seen in DS9), dy = 3 is to shift the image "UP".
            dy (float): shift distance (in pixel) along y (vertical). 
                Note that elements in one row has the same y but different x. 
                Example: dx = 2 is to shift the mask "RIGHT" (as seen in DS9), dy = 3 is to shift the image "UP".
            method (str): interpolation method. Use 'lanczos' or 'iraf'. 
                If using 'iraf', default interpolation is 'poly3. 'Lanczos' requires ``GalSim`` installed.
            order (int): the order of Lanczos interpolation (>0).
            cval (float): value to fill the edges. Default is 0.

        Returns:
            shift_mask (ndarray): shifted mask. The "mask" attribute of ``Celestial`` class will also be changed accordingly.
        '''

        ny, nx = self.mask.shape
        if abs(dx) > nx or abs(ny) > ny:
            raise ValueError('# Shift distance is beyond the image size.')
        if method == 'lanczos':
            try: # try to import galsim
                from galsim import degrees, Angle
                from galsim.interpolant import Lanczos
                from galsim import Image, InterpolatedImage
                from galsim.fitswcs import AstropyWCS
            except:
                raise ImportError('# Import ``galsim`` failed! Please check if ``galsim`` is installed!')
            # Begin shift
            assert (order > 0) and isinstance(order, int), 'order of ' + method + ' must be positive interger.'
            galimg = InterpolatedImage(Image(self.mask, dtype=float), 
                                    scale=self.pixel_scale, x_interpolant=Lanczos(order))
            galimg = galimg.shift(dx=dx * self.pixel_scale, dy=dy * self.pixel_scale)
            result = galimg.drawImage(scale=self.pixel_scale, nx=nx, ny=ny)#, wcs=AstropyWCS(self.wcs))
            self._mask = result.array
            # Change the WCS of image
            hdr = copy.deepcopy(self.header)
            hdr['CRPIX1'] += dx
            hdr['CRPIX2'] += dy
            self.header = hdr
            self.wcs = wcs.WCS(hdr)
            self._wcs_header_merge()
            return result.array
        elif method == 'iraf':
            self.save_to_fits('./_temp.fits', 'mask')
            imshift('./_temp.fits', './_shift_temp.fits', dx, dy, interp_type='poly3', boundary_type='constant')
            hdu = fits.open('./_shift_temp.fits')
            self.mask = hdu[0].data
            self.shape = hdu[0].data.shape
            self.header = hdu[0].header
            self.wcs = wcs.WCS(self.header)
            hdu.close()
            imdelete('./*temp.fits')
        else:
            raise ValueError("# Not supported interpolation method. Use 'lanczos' or 'iraf'.")

    def shift_Celestial(self, dx, dy, method='iraf', order=5, cval=0.0):
        '''Shift the Celestial object, including image and mask.

        Parameters:
            dx (float): shift distance (in pixel) along x (horizontal). 
                Note that elements in one row has the same y but different x. 
                Example: dx = 2 is to shift the image "RIGHT" (as seen in DS9), dy = 3 is to shift the image "UP".
            dy (float): shift distance (in pixel) along y (vertical). 
                Note that elements in one row has the same y but different x. 
                Example: dx = 2 is to shift the image "RIGHT" (as seen in DS9), dy = 3 is to shift the image "UP".
            method (str): interpolation method. Use 'lanczos' or 'iraf'. 
                If using 'iraf', default interpolation is 'poly3. 'Lanczos' requires ``GalSim`` installed.
            order (int): the order of Lanczos interpolation (>0).
            cval (float): value to fill the edges. Default is 0.
        
        Returns:
            None
        '''
        self.shift_image(dx, dy, method=method, order=order, cval=cval)
        if hasattr(self, 'mask'):
            self.shift_mask(dx, dy, method=method, order=order, cval=cval)
    
    def resize_image(self, f, method='iraf', order=5, cval=0.0):
        '''
        Zoom/Resize the image of Celestial object. 
        f > 1 means the image will be resampled (finer)! f < 1 means the image will be degraded.

        Parameters:
            f (float): the positive factor of zoom. If 0 < f < 1, the image will be resized to smaller one.
            method (str): interpolation method. Use 'lanczos' or 'iraf'. 'Lanczos' requires ``GalSim`` installed.
            order (int): the order Lanczos interpolation (>0).
            cval (float): value to fill the edges. Default is 0.

        Returns:
            resize_image (ndarray): resized image. The "image" attribute of ``Celestial`` class will also be changed accordingly.
        '''

        if method == 'lanczos':
            try: # try to import galsim
                from galsim import degrees, Angle
                from galsim.interpolant import Lanczos
                from galsim import Image, InterpolatedImage
                from galsim.fitswcs import AstropyWCS
            except:
                raise ImportError('# Import ``galsim`` failed! Please check if ``galsim`` is installed!')

            assert (order > 0) and isinstance(order, int), 'order of ' + method + ' must be positive interger.'
            galimg = InterpolatedImage(Image(self.image, dtype=float), 
                                    scale=self.pixel_scale, x_interpolant=Lanczos(order))
            #galimg = galimg.magnify(f)
            ny, nx = self.image.shape
            result = galimg.drawImage(scale=self.pixel_scale / f, nx=round(nx * f), ny=round(ny * f))#, wcs=AstropyWCS(self.wcs))
            self.wcs = self._resize_wcs(self.image, self.wcs, f)
            self._image = result.array
            self.shape = self.image.shape
            self._wcs_header_merge()
            self.pixel_scale /= f
            return result.array
        elif method == 'iraf':
            self.save_to_fits('./_temp.fits', 'image')
            if f > 1:
                magnify('./_temp.fits', './_resize_temp.fits', f, f)
            else:
                blkavg('./_temp.fits', './_resize_temp.fits', 1/f, 1/f, option='sum')
            hdu = fits.open('./_resize_temp.fits')
            self.image = hdu[0].data
            self.shape = hdu[0].data.shape
            self.header = hdu[0].header
            self.wcs = wcs.WCS(self.header)
            self.pixel_scale /= f
            hdu.close()
            imdelete('./*temp.fits')
        else:
            raise ValueError("# Not supported interpolation method. Use 'lanczos' or 'iraf'.")
        
    def resize_mask(self, f, method='iraf', order=5, cval=0.0):
        '''
        Zoom/Resize the mask of Celestial object. 
        f > 1 means the mask will be resampled (finer)! f < 1 means the mask will be degraded.

        Parameters:
            f (float): the positive factor of zoom. If 0 < f < 1, the mask will be resized to smaller one.
            method (str): interpolation method. Use 'lanczos' or 'spline' or 'iraf'. 'Lanczos' requires ``GalSim`` installed.
            order (int): the order Lanczos interpolation (>0).
            cval (float): value to fill the edges. Default is 0.

        Returns:
            resize_mask (ndarray): resized mask. The "mask" attribute of ``Celestial`` class will also be changed accordingly.
        '''

        if method == 'lanczos':
            try: # try to import galsim
                from galsim import degrees, Angle
                from galsim.interpolant import Lanczos
                from galsim import Image, InterpolatedImage
                from galsim.fitswcs import AstropyWCS
            except:
                raise ImportError('# Import ``galsim`` failed! Please check if ``galsim`` is installed!')

            assert (order > 0) and isinstance(order, int), 'order of ' + method + ' must be positive interger.'
            galimg = InterpolatedImage(Image(self.mask, dtype=float), 
                                    scale=self.pixel_scale, x_interpolant=Lanczos(order))
            ny, nx = self.mask.shape
            result = galimg.drawImage(scale=self.pixel_scale / f, nx=round(nx * f), ny=round(ny * f))#, wcs=AstropyWCS(self.wcs))
            self.wcs = self._resize_wcs(self.image, self.wcs, f)
            self._image = result.array
            self.shape = self.mask.shape
            self._wcs_header_merge()
            self.pixel_scale /= f
            return result.array
        elif method == 'iraf':
            self.save_to_fits('./_temp.fits', 'mask')
            if f > 1:
                magnify('./_temp.fits', './_resize_temp.fits', f, f)
            else:
                blkavg('./_temp.fits', './_resize_temp.fits', 1/f, 1/f, option='sum')
            hdu = fits.open('./_resize_temp.fits')
            self.mask = hdu[0].data
            self.shape = hdu[0].data.shape
            self.header = hdu[0].header
            self.wcs = wcs.WCS(self.header)
            self.pixel_scale /= f
            hdu.close()
            imdelete('./*temp.fits')
        else:
            raise ValueError("# Not supported interpolation method. Use 'lanczos' or 'iraf'.")

    def resize_Celestial(self, f, method='iraf', order=5, cval=0.0):
        '''
        Resize the Celestial object, including both image and mask.
        f > 1 means the image/mask will be resampled! f < 1 means the image/mask will be degraded.

        Parameters:
            f (float): the positive factor of zoom. If 0 < f < 1, the mask will be resized to smaller one.
            method (str): interpolation method. Use 'lanczos' or 'spline' or 'iraf'. 'Lanczos' requires ``GalSim`` installed.
            order (int): the order Lanczos interpolation (>0).
            cval (float): value to fill the edges. Default is 0.
        
        Returns:
            None
        '''
        self.resize_image(f, method=method, order=order, cval=cval)
        if hasattr(self, 'mask'):
            self.resize_mask(f, method=method, order=order, cval=cval)

    # Display image/mask
    def display_image(self, **kwargs):
        """
        Take a peek at the image, using "zscale", "arcsinh" streching and "viridis" colormap. You can change them by adding **kwargs.

        Parameters:
            **kwargs: arguments in ``mrf.display.display_single``.

        Returns:
            None
        """
        display_single(self.image, scale_bar_length=self.scale_bar_length, **kwargs)

    def display_mask(self, **kwargs):
        """
        Take a peek at the mask.

        Parameters:
            **kwargs: arguments in ``mrf.display.display_single``.
            
        Returns:
            None
        """
        display_single(self.mask, scale='linear', 
                        cmap=SEG_CMAP, scale_bar_length=self.scale_bar_length, **kwargs)

    def display_Celestial(self, **kwargs):
        """
        Take a peek at the masked image, using "zscale", "arcsinh" streching and "viridis" colormap. You can change them by adding **kwargs.

        Parameters:
            **kwargs: arguments in ``mrf.display.display_single``.
            
        Returns:
            None
        """
        if hasattr(self, 'mask'):
            display_single(self.image * (~self.mask.astype(bool)), 
                            scale_bar_length=self.scale_bar_length, **kwargs)
        else:
            self.display_image()

class Star(Celestial):
    """
    This ``Star`` class is the inheritance of ``Celestial`` class. 
    It represents a small cutout, which is typically a star. 
    Other than the functions inherited from ``Celestial``, ``Star`` object has extra functions such as ``centralize``, ``mask_out_contam``.
    """
    def __init__(self, img, header, starobj, colnames=['x', 'y'], halosize=40, padsize=40, mask=None, hscmask=None):
        """
        Initialize ``Star`` object. 
        
        Parameters:
            img (numpy 2-D array): the image from which the cutout of star is made.
            header: header of image, containing WCS information. Typically it is ``astropy.io.fits.header`` object.
            starobj: A row of ``astropy.table.Table``, containing basic information of the star, such as ``ra``, `dec`` and magnitudes.
            colnames (list of str): indicating the columns which contains position of the star. It could be ['x', 'y'] or ['ra', 'dec'].
            halosize (float): the radial size of cutout. If ``halosize=40``, the square cutout will be 80 * 80 pix.
            padsize (float): The image will be padded in order to make cutout of stars near the edge of input image. 
                ``padsize`` should be equal to or larger than ``halosize``.
            mask (numpy 2-D array): the mask of input big image.
            hscmask (numpy 2-D array): the hscmask of input image.
        
        Returns:
            None
        """
        Celestial.__init__(self, img, mask, header=header)
        if hscmask is not None:
            self.hscmask = hscmask
        self.name = 'star'
        self.scale_bar_length = 3

        # Trim the image to star size   
        # starobj should at least contain x, y, (or ra, dec)
        if 'x' in colnames or 'y' in colnames:
            # Position of a star, in numpy convention
            x_int = int(starobj['x'])
            y_int = int(starobj['y'])
            dx = -1.0 * (starobj['x'] - x_int)
            dy = -1.0 * (starobj['y'] - y_int)
        elif 'ra' in colnames or 'dec' in colnames:
            w = self.wcs
            x, y = w.wcs_world2pix(starobj['ra'], starobj['dec'], 0)
            x_int = int(x)
            y_int = int(y)
            dx = -1.0 * (x - x_int)
            dy = -1.0 * (y - y_int)

        halosize = int(halosize)
        # Make padded image to deal with stars near the edges
        padsize = int(padsize)
        ny, nx = self.image.shape
        im_padded = np.zeros((ny + 2 * padsize, nx + 2 * padsize))
        im_padded[padsize: ny + padsize, padsize: nx + padsize] = self.image
        # Star itself, but no shift here.
        halo = im_padded[y_int + padsize - halosize: y_int + padsize + halosize + 1, 
                         x_int + padsize - halosize: x_int + padsize + halosize + 1]
        self._image = halo
        self.shape = halo.shape
        self.cen_xy = [x_int, y_int]
        self.dx = dx
        self.dy = dy   
        # FLux
        self.flux = starobj['flux']
        self.fluxann = starobj['flux_ann']
        self.fluxauto = starobj['flux_auto']

        if hasattr(self, 'mask'):
            im_padded = np.zeros((ny + 2 * padsize, nx + 2 * padsize))
            im_padded[padsize: ny + padsize, padsize: nx + padsize] = self.mask
            # Mask itself, but no shift here.
            halo = (im_padded[y_int + padsize - halosize: y_int + padsize + halosize + 1, 
                                x_int + padsize - halosize: x_int + padsize + halosize + 1])
            self._mask = halo
        
        if hasattr(self, 'hscmask'):
            im_padded = np.zeros((ny + 2 * padsize, nx + 2 * padsize))
            im_padded[padsize: ny + padsize, padsize: nx + padsize] = self.hscmask
            # Mask itself, but no shift here.
            halo = (im_padded[y_int + padsize - halosize: y_int + padsize + halosize + 1, 
                                x_int + padsize - halosize: x_int + padsize + halosize + 1])
            self.hscmask = halo

    def centralize(self, method='iraf', order=5, cval=0.0):
        """
        Shift the cutout to the true position of the star using interpolation. 

        Parameters:
            method (str): interpolation method. Options are "iraf" and "lanczos". "Lanczos" requires ``GalSim`` installed.
            order (int): the order of Lanczos interpolation (>0).
            cval (float): value to fill the edges. Default is 0.
        Returns:
            None
        """

        self.shift_Celestial(self.dx, self.dy, method=method, order=order, cval=cval)

    def sub_bkg(self, sigma=4.5, deblend_cont=0.0001, verbose=True):
        """
        Subtract the locally-measured background of ``Star`` object. The sky is measured by masking out objects using ``sep``.
        Be cautious and be aware what you do when using this function.

        Parameters:
            sigma (float): The sigma in ``SExtractor``.
            deblend_cont (float): Deblending parameter.
            verbose (bool): Whether print out background value.
        
        Returns:
            None
        """
        # Actually this should be estimated in larger cutuouts.
        # So make another cutout (larger)!
        from astropy.convolution import convolve, Box2DKernel
        from .image import extract_obj, seg_remove_cen_obj
        from sep import Background
        img_blur = convolve(abs(self.image), Box2DKernel(2))
        img_objects, img_segmap = extract_obj(abs(img_blur), b=10, f=4, sigma=sigma, minarea=2, pixel_scale=self.pixel_scale,
                                                deblend_nthresh=32, deblend_cont=deblend_cont, 
                                                sky_subtract=False, show_fig=False, verbose=False)
        bk = Background(self.image, img_segmap != 0)
        glbbck = bk.globalback
        self.globalback = glbbck
        if verbose:
            print('# Global background: ', glbbck)
        self.image -= glbbck

    def get_masked_image(self, cval=np.nan):
        """
        Mask image according to the mask.

        Parameter:
            cval: value to fill the void. Default is NaN, but sometimes NaN is problematic. 
        
        Return:
            imgcp (numpy 2-D array): masked image.
        """

        if not hasattr(self, 'mask'):
            print("This ``Star`` object doesn't have a ``mask``!")
            return self.image
        else:
            imgcp = copy.copy(self.image)
            imgcp[self.mask.astype(bool)] = cval
            return imgcp

    def mask_out_contam(self, sigma=4.5, deblend_cont=0.0005, blowup=True, show_fig=True, verbose=True):
        """
        Mask out contamination in the cutout of star. Contamination may be stars, galaxies or artifacts. 
        This function uses ``sep`` to identify and mask contamination。

        Parameters:
            sigma (float): The sigma in ``SExtractor``. Default is 4.5.
            deblend_cont (float): Deblending parameter. Default is 0.0005.
            blowup (bool): Whether blow up the segmentation mask by convolving a 1.5 pixel Gaussian kernel.
            show_fig (bool): Whether show the figure.
            verbose (bool): Whether print out results.

        Returns:
            None
        """
        
        from astropy.convolution import convolve, Box2DKernel
        from .utils import extract_obj, seg_remove_cen_obj
        img_blur = convolve(abs(self.image), Box2DKernel(2))
        img_objects, img_segmap = extract_obj(abs(img_blur), b=5, f=4, sigma=sigma, minarea=2, pixel_scale=self.pixel_scale,
                                                deblend_nthresh=32, deblend_cont=deblend_cont, 
                                                sky_subtract=False, show_fig=show_fig, verbose=verbose)
        # remove central object from segmap
        img_segmap = seg_remove_cen_obj(img_segmap) 
        detect_mask = (img_segmap != 0).astype(float)
        if blowup is True:
            from astropy.convolution import convolve, Gaussian2DKernel
            cv = convolve(detect_mask, Gaussian2DKernel(1.5))
            detect_mask = (cv > 0.1).astype(float)
        self.mask = detect_mask
        return 
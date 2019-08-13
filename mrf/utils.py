import os
import sep
import copy
import scipy
import numpy as np
import matplotlib.pyplot as plt

from astropy import wcs
from astropy.io import fits
from astropy.table import Table, Column
from astropy.stats import sigma_clip
from astropy.nddata.utils import block_replicate, block_reduce
from photutils import CosineBellWindow, TopHatWindow

from .display import display_single, SEG_CMAP

USNO_vizier = 'I/252/out'
APASS_vizier = 'II/336'

#########################################################################
########################## General Tools  ###############################
#########################################################################

# Calculate physical size of a given redshift
def phys_size(redshift, is_print=True, H0=70, Omegam=0.3):
    ''' Calculate the corresponding physical size per arcsec of a given redshift
        in the Lambda-CDM cosmology.

    Requirement:
    -----------
    astropy
    
    Parameters:
    -----------
    redshift: float
    is_print: boolean. If true, it will print out the physical scale at the given redshift.
    Omegam: float, density parameter of matter. It should be within [0, 1]. 

    Returns:
    -----------
    physical_size: float, in 'kpc/arcsec'
    '''
    from astropy.cosmology import FlatLambdaCDM
    cosmos = FlatLambdaCDM(H0=H0, Om0=Omegam) 
    physical_size = 1 / cosmos.arcsec_per_kpc_comoving(0.1).value # kpc/arcsec
    if is_print:
        print ('At redshift', redshift, ', 1 arcsec =', physical_size, 'kpc')
    return physical_size

# Cutout image
def img_cutout(img, wcs, coord_1, coord_2, size=60.0, pix=2.5,
               prefix='img_cutout', pixel_unit=False, img_header=None, 
               out_dir=None, save=True):
    """(From kungpao) Generate image cutout with updated WCS information.
    ----------
    Parameters:
        img: 2d array.
        pixel_unit: boolen, optional
                    When True, coord_1, cooord_2 becomes X, Y pixel coordinates.
                    Size will also be treated as in pixels.
        wcs: astropy wcs object of the input image.
        coord_1: ra or x of the center.
        coord_2: dec or y of the center.
        size: image size, default in arcsec unit.
        pix: pixel size.
        img_header: the astropy header object of the input image. 
                    In case you can save the infomation in this header to the new header.
    """
    from astropy.nddata import Cutout2D
    if not pixel_unit:
        # imgsize in unit of arcsec
        cutout_size = np.asarray(size) / pix
        cen_x, cen_y = wcs.wcs_world2pix(coord_1, coord_2, 0)
    else:
        cutout_size = np.asarray(size)
        cen_x, cen_y = coord_1, coord_2

    cen_pos = (int(cen_x), int(cen_y))
    dx = -1.0 * (cen_x - int(cen_x))
    dy = -1.0 * (cen_y - int(cen_y))

    # Generate cutout
    cutout = Cutout2D(img, cen_pos, cutout_size, wcs=wcs)

    # Update the header
    cutout_header = cutout.wcs.to_header()
    if img_header is not None:
        intersect = [k for k in img_header if k not in cutout_header]
        for keyword in intersect:
            cutout_header.set(keyword, img_header[keyword], img_header.comments[keyword])
    
    if 'PC1_1' in dict(cutout_header).keys():
        cutout_header['CD1_1'] = cutout_header['PC1_1']
        #cutout_header['CD1_2'] = cutout_header['PC1_2']
        #cutout_header['CD2_1'] = cutout_header['PC2_1']
        cutout_header['CD2_2'] = cutout_header['PC2_2']
        cutout_header['CDELT1'] = cutout_header['CD1_1']
        cutout_header['CDELT2'] = cutout_header['CD2_2']
        cutout_header.pop('PC1_1')
        #cutout_header.pop('PC2_1')
        #cutout_header.pop('PC1_2')
        cutout_header.pop('PC2_2')
        #cutout_header.pop('CDELT1')
        #cutout_header.pop('CDELT2')
    
    # Build a HDU
    hdu = fits.PrimaryHDU(header=cutout_header)
    hdu.data = cutout.data
    #hdu.data = np.flipud(cutout.data)
    # Save FITS image
    if save:
        fits_file = prefix + '.fits'
        if out_dir is not None:
            fits_file = os.path.join(out_dir, fits_file)

        hdu.writeto(fits_file, overwrite=True)

    return cutout, [cen_pos, dx, dy], cutout_header

# Save 2-D numpy array to `fits`
def save_to_fits(img, fits_file, wcs=None, header=None, overwrite=True):
    """Save numpy 2-D arrays to `fits` file. (from `kungpao`)
    Parameters:
        img (np.array, 2d): The 2-D array to be saved
        fits_file (str): File name of `fits` file
        wcs (astropy.wcs.WCS class): World coordinate system of this image
        header (astropy.io.fits.header or str): header of this image
        overwrite (bool): Default is True

    Returns:
        None
    """
    img_hdu = fits.PrimaryHDU(img)

    if header is not None:
        img_hdu.header = header
        if wcs is not None:
            hdr = copy.deepcopy(header)
            wcs_header = wcs.to_header()
            import fnmatch
            for i in hdr:
                if i in wcs_header:
                    hdr[i] = wcs_header[i]
                if fnmatch.fnmatch(i, 'CD?_?'):
                    hdr[i] = wcs_header['PC' + i.lstrip('CD')]
            img_hdu.header = hdr
    elif wcs is not None:
        wcs_header = wcs.to_header()
        img_hdu.header = wcs_header

    else:
        img_hdu = fits.PrimaryHDU(img)

    if os.path.islink(fits_file):
        os.unlink(fits_file)

    img_hdu.writeto(fits_file, overwrite=overwrite)
    return img_hdu

def seg_remove_cen_obj(seg):
    """Remove the central object from the segmentation.
    Parameters:
        seg (numpy 2-D array): segmentation map

    Returns:
        seg_copy (numpy 2-D array): the segmentation map with central object removed
    """
    seg_copy = copy.deepcopy(seg)
    seg_copy[seg == seg[int(seg.shape[0] / 2.0), int(seg.shape[1] / 2.0)]] = 0

    return seg_copy

def mask_remove_cen_obj(mask):
    """Remove the central object from the binary 0-1 mask.
    Parameters:
        mask (numpy 2-D array): binary mask

    Returns:
        mask_copy (numpy 2-D array): a mask with central object removed
    """
    from scipy.ndimage import label
    mask_copy = copy.deepcopy(mask)
    seg = label(mask)[0]
    mask_copy[seg == seg[int(seg.shape[0] / 2.0), int(seg.shape[1] / 2.0)]] = 0

    return mask_copy

def seg_remove_obj(seg, x, y):
    """Remove an object from the segmentation given its coordinate.
        
    Parameters:
        seg (numpy 2-D array): segmentation mask
        x, y (int): coordinates.
    Returns:
        seg_copy (numpy 2-D array): the segmentation map with certain object removed
    """
    seg_copy = copy.deepcopy(seg)
    seg_copy[seg == seg[int(y), int(x)]] = 0

    return seg_copy

def mask_remove_obj(mask, x, y):
    """Remove an object from the mask given its coordinate.
        
    Parameters:
        mask (numpy 2-D array): binary mask
        x, y (int): coordinates.
    Returns:
        mask_copy (numpy 2-D array): the mask with certain object removed
    """
    from scipy.ndimage import label
    mask_copy = copy.deepcopy(mask)
    seg = label(mask)[0]
    mask_copy[seg == seg[int(y), int(x)]] = 0

    return mask_copy

def img_replace_with_noise(img, mask):
    """ This function add Gaussian noise to the masked region. 
        The characteristics of Gaussian is decided by `sep` locally measured sky value and stderr.
        (stole from `kungpao`)

    Parameters:
        img (numpy 2-D array): image.
        mask (numpy 2-D array): pixels that you want to mask out should have value of ONE.

    Returns:
        img_noise_replace (numpy 2-D array): image with mask replaced by noise
    """
    import sep
    import copy
    if sep.__version__ < '0.8.0':
        raise ImportError('Please update `sep` to most recent version! Current version is ' + sep.__version__)
    else:
        bkg = sep.Background(img, mask=(mask.astype(bool)))
        sky_noise_add = np.random.normal(loc=bkg.back(), 
                                        scale=bkg.rms(), 
                                        size=img.shape)
        img_noise_replace = copy.deepcopy(img)
        img_noise_replace[mask != 0] = sky_noise_add[mask != 0]
        return img_noise_replace

def circularize(img, n=14, print_g=True):
    """ Circularize an image. Inspired by http://adsabs.harvard.edu/abs/2011PASP..123.1218A.
    Parameters:
        img (numpy 2-D array): image
        n (int): times of circularization. For example, the output image 
            will be invariant under rotation `theta = 360 / 2^14` if `n=14`.
        print_g (bool): if true, a parameter describing the asymmetricness of input image. 
            `g=0` means perfectly symmetric.
    Returns:
        img_cir (numpy 2-D array): circularized image.
    """
    from scipy.ndimage.interpolation import rotate
    a = img
    for i in range(n):
        theta = 360 / 2**(i + 1)
        if i == 0:
            temp = a
        else:
            temp = b
        b = rotate(temp, theta, order=3, mode='constant', cval=0.0, reshape=False)
        c = .5 * (a + b)
        a = b
        b = c 

    img_cir = b
    if print_g is True:
        print('The asymmetry parameter g of given image is ' + 
                str(abs(np.sum(img_cir - img))))
    return img_cir

def azimuthal_average(image, center=None, stddev=True, binsize=0.5, interpnan=False):
    """
    Modified based on https://github.com/keflavich/image_tools/blob/master/image_tools/radialprofile.py
    Calculate the azimuthally averaged radial profile.
    
    Parameters:
        imgae (numpy ndarray): 2-D image
        center (list): [x, y] pixel coordinates. If None, use image center.
            Note that x is horizontal and y is vertical, y, x = image.shape.
        stdev (bool): if True, the stdev of profile will also be returned.
        binsize (float): size of the averaging bin. Can lead to strange results if
            non-binsize factors are used to specify the center and the binsize is
            too large.
        interpnan (bool): Interpolate over NAN values, i.e. bins where there is no data?
    
    Returns:
        If `stdev == True`, it will return [radius, profile, stdev]; 
        else, it will return [radius, profile].
    """
    # Calculate the indices from the image
    y, x = np.indices(image.shape)

    if center is None:
        center = np.array([(x.max() - x.min()) / 2.0, (y.max() - y.min()) / 2.0])

    r = np.hypot(x - center[0], y - center[1])
    
    # The 'bins' as initially defined are lower/upper bounds for each bin
    # so that values will be in [lower,upper)  
    nbins = int(np.round(r.max() / binsize) + 1)
    maxbin = nbins * binsize
    bins = np.linspace(0, maxbin, nbins + 1)
    # We're probably more interested in the bin centers than their left or right sides...
    bin_centers = (bins[1:] + bins[:-1]) / 2.0

    # There are never any in bin 0, because the lowest index returned by digitize is 1
    nr = np.histogram(r, bins)[0] # nr is how many pixels are within each bin

    # Radial profile itself
    nan_flag = np.isnan(image) # get rid of nan
    #profile = np.histogram(r, bins, weights=image)[0] / nr
    profile = np.histogram(r[~nan_flag], bins, weights=image[~nan_flag])[0] / nr
    
    if interpnan:
        profile = np.interp(bin_centers, bin_centers[~np.isnan(profile)],
                            profile[~np.isnan(profile)])
    if stddev:
        # Find out which radial bin each point in the map belongs to
        # recall that bins are from 1 to nbins
        whichbin = np.digitize(r.ravel(), bins)
        profile_std = np.array([np.nanstd(image.ravel()[whichbin == b]) for b in range(1, nbins + 1)])
        profile_std /= np.sqrt(nr) # 均值的偏差
        return [bin_centers, profile, profile_std]
    else:
        return [bin_centers, profile]

    """
    Calculate the azimuthally averaged radial profile.

    image - The 2D image
    center - The [x,y] pixel coordinates used as the center. The default is 
             None, which then uses the center of the image (including 
             fracitonal pixels).
    
    """
    # Calculate the indices from the image
    y, x = np.indices(image.shape)

    if not center:
        center = np.array([(x.max()-x.min())/2.0, (x.max()-x.min())/2.0])

    r = np.hypot(x - center[0], y - center[1])

    # Get sorted radii
    ind = np.argsort(r.flat)
    r_sorted = r.flat[ind]
    i_sorted = image.flat[ind]

    # Get the integer part of the radii (bin size = 1)
    r_int = r_sorted.astype(int)

    # Find all pixels that fall within each radial bin.
    deltar = r_int[1:] - r_int[:-1]  # Assumes all radii represented
    rind = np.where(deltar)[0]       # location of changed radius
    nr = rind[1:] - rind[:-1]        # number of radius bin
    
    # Cumulative sum to figure out sums for each radius bin
    csim = np.cumsum(i_sorted, dtype=float)
    tbin = csim[rind[1:]] - csim[rind[:-1]]

    radial_prof = tbin / nr

    return radial_prof


#########################################################################
####################### Source extract related ##########################
#########################################################################

# evaluate_sky objects for a given image
def extract_obj(img, b=64, f=3, sigma=5, pixel_scale=0.168, minarea=5, 
    deblend_nthresh=32, deblend_cont=0.005, clean_param=1.0, 
    sky_subtract=False, flux_auto=True, flux_aper=None, show_fig=True, verbose=True):
    '''Extract objects for a given image, using `sep`. This is from `slug`.

    Parameters:
    ----------
    img: 2-D numpy array
    b: float, size of box
    f: float, size of convolving kernel
    sigma: float, detection threshold
    pixel_scale: float, default is 0.168 (HSC pixel size)
    minarea: float, minimum number of connected pixels
    deblend_nthresh: float, Number of thresholds used for object deblending
    deblend_cont: float, Minimum contrast ratio used for object deblending. Set to 1.0 to disable deblending. 
    clean_param: float, Cleaning parameter (see SExtractor manual)
    sky_subtract: bool, whether subtract sky before extract objects (this will affect the measured flux).
    flux_auto: bool, whether return AUTO photometry (see SExtractor manual)
    flux_aper: list, such as [3, 6], which gives flux within [3 pix, 6 pix] annulus.

    Returns:
    -------
    objects: `astropy` Table, containing the positions,
        shapes and other properties of extracted objects.
    segmap: 2-D numpy array, segmentation map
    '''

    # Subtract a mean sky value to achieve better object detection
    b = 30  # Box size
    f = 5   # Filter width
    bkg = sep.Background(img, bw=b, bh=b, fw=f, fh=f)
    data_sub = img - bkg.back()
    
    sigma = sigma
    if sky_subtract:
        input_data = data_sub
    else:
        input_data = img
    objects, segmap = sep.extract(input_data,
                                  sigma,
                                  err=bkg.globalrms,
                                  segmentation_map=True,
                                  filter_type='matched',
                                  deblend_nthresh=deblend_nthresh,
                                  deblend_cont=deblend_cont,
                                  clean=True,
                                  clean_param=clean_param,
                                  minarea=minarea)
    if verbose:                              
        print("# Detect %d objects" % len(objects))
    objects = Table(objects)
    objects.add_column(Column(data=np.arange(len(objects)) + 1, name='index'))
    # Maximum flux, defined as flux within six 'a' in radius.
    objects.add_column(Column(data=sep.sum_circle(input_data, objects['x'], objects['y'], 
                                    6. * objects['a'])[0], name='flux_max'))
    # Add FWHM estimated from 'a' and 'b'. 
    # This is suggested here: https://github.com/kbarbary/sep/issues/34
    objects.add_column(Column(data=2* np.sqrt(np.log(2) * (objects['a']**2 + objects['b']**2)), 
                              name='fwhm_custom'))
    
    # Use Kron radius to calculate FLUX_AUTO in SourceExtractor.
    # Here PHOT_PARAMETER = 2.5, 3.5
    if flux_auto:
        kronrad, krflag = sep.kron_radius(input_data, objects['x'], objects['y'], 
                                          objects['a'], objects['b'], 
                                          objects['theta'], 6.0)
        flux, fluxerr, flag = sep.sum_circle(input_data, objects['x'], objects['y'], 
                                            2.5 * (kronrad), subpix=1)
        flag |= krflag  # combine flags into 'flag'

        r_min = 1.75  # minimum diameter = 3.5
        use_circle = kronrad * np.sqrt(objects['a'] * objects['b']) < r_min
        cflux, cfluxerr, cflag = sep.sum_circle(input_data, objects['x'][use_circle], objects['y'][use_circle],
                                                r_min, subpix=1)
        flux[use_circle] = cflux
        fluxerr[use_circle] = cfluxerr
        flag[use_circle] = cflag
        objects.add_column(Column(data=flux, name='flux_auto'))
        objects.add_column(Column(data=kronrad, name='kron_rad'))
        
    if flux_aper is not None:
        if len(flux_aper) != 2:
            raise ValueError('"flux_aper" must be a list with length = 2.')
        objects.add_column(Column(data=sep.sum_circle(input_data, objects['x'], objects['y'], flux_aper[0])[0], 
                                  name='flux_aper_1'))
        objects.add_column(Column(data=sep.sum_circle(input_data, objects['x'], objects['y'], flux_aper[1])[0], 
                                  name='flux_aper_2')) 
        objects.add_column(Column(data=sep.sum_circann(input_data, objects['x'], objects['y'], 
                                       flux_aper[0], flux_aper[1])[0], name='flux_ann'))

    # plot background-subtracted image
    if show_fig:
        fig, ax = plt.subplots(1,2, figsize=(12,6))

        ax[0] = display_single(data_sub, ax=ax[0], scale_bar_length=60, pixel_scale=pixel_scale)
        from matplotlib.patches import Ellipse
        # plot an ellipse for each object
        for obj in objects:
            e = Ellipse(xy=(obj['x'], obj['y']),
                        width=8*obj['a'],
                        height=8*obj['b'],
                        angle=obj['theta'] * 180. / np.pi)
            e.set_facecolor('none')
            e.set_edgecolor('red')
            ax[0].add_artist(e)
        ax[1] = display_single(segmap, scale='linear', cmap=SEG_CMAP , ax=ax[1])
        plt.savefig('./extract_obj.png', bbox_inches='tight')
    return objects, segmap

# Detect sources and make flux map
def Flux_Model(img, header, b=64, f=3, sigma=2.5, minarea=3, 
                deblend_cont=0.005, deblend_nthresh=32, save=False, output_suffix='flux_model'):
    """ Extract sources from given image and return a flux map (not segmentation map).
        The flux map will be saved as '_flux_' + output_suffix + '.fits', along with segmentation map.

    Parameters:
        img (numpy 2-D array): Image itself
        header: the header of the image
        sigma (float): We detect objects above this sigma
        minarea (float): minimum area of the object, in pixels
        deblend_cont (float): Minimum contrast ratio used for object deblending. Default is 0.005. 
            To entirely disable deblending, set to 1.0.
        deblend_nthresh (float): Number of thresholds used for object deblending. Default is 32.
        output_suffix (str): Suffix of output image and segmentation map.

    Returns:
        objects (astropy.table.Table class): Table of detected objects.
        segmap (numpy 2-D array): Segmentation map.
        im_fluxes (numpy 2-D array): Flux map.
    """

    objects, segmap = extract_obj(img, b=b, f=f, sigma=sigma, minarea=minarea, show_fig=False,
                                  deblend_nthresh=deblend_nthresh, deblend_cont=deblend_cont, 
                                  sky_subtract=False)
    im_seg = segmap
    galid = objects['index'].data.astype(np.float)
    flux = objects['flux_auto'].data.astype(np.float)
    im_fluxes = im_seg.astype(float)
    im_seg_slice_ind = np.where(im_seg > 0) # The (x,y) index of non-zero pixels in segmap
    im_seg_slice = im_seg[im_seg > 0] # The index of non-zero pixels in segmap
    im_fluxes_slice = im_fluxes[im_seg > 0] # The fluxes of non-zero pixels in segmap

    for i in range(len(objects)):
        ind = np.where(np.isin(im_seg_slice, galid[i]))
        im_fluxes_slice[ind] = flux[i]

    im_fluxes[im_seg_slice_ind] = im_fluxes_slice # Change the objid to flux of this obj
    if save:
        save_to_fits(im_fluxes, '_flux_' + output_suffix + '.fits', header=header)
        save_to_fits(segmap, '_seg' + output_suffix + '.fits', header=header)
    return objects, segmap, im_fluxes

# Simply remove stars by masking them out
def query_star(img, header, method='gaia', bright_lim=15.5, catalog_dir=None):
    """ 
    Parameters:
        img (2-D numpy array): image itselt.
        header: the header of this image.
        method (str): here three methods are provided: 'gaia', 'apass' or 'usno'.
        bright_lim (float): the magnitude limit of stars to be masked out. 
        catalog_dir (str): optional, you can provide local catalog here.

    Returns:
        star_cat
    """
    if method.lower() == 'gaia':
        from kungpao import imtools, query
        from astropy import wcs
        print('### Querying Gaia Data ###')
        gaia_stars, gaia_mask = imtools.gaia_star_mask(img, wcs.WCS(header), 
                                                       pix=header['CD2_2'] * 3600, 
                                                       size_buffer=4, gaia_bright=bright_lim, 
                                                       factor_f=2.0, factor_b=1.2)
        #gaia_stars = gaia_stars[gaia_stars['phot_bp_mean_mag'] < bright_lim]                                               
        return gaia_stars
    elif method.lower() == 'apass' or method.lower() == 'usno':
        if catalog_dir is not None: # catalog is provided
            print("You provided star catalog file!")
            # Read catalog directory
            _, file_extension = os.path.splitext(catalog_dir)
            if file_extension.lower() == 'fits':
                catalog = Table.read(catalog_dir, format='fits')
            else:
                catalog = Table.read(catalog_dir, format='ascii')
        else: # Online query!
            print("### Online querying " + method.upper() + " data from VizieR. ###")
            # Construct query
            from astropy.coordinates import SkyCoord
            import astropy.units as u
            from astropy import wcs
            w = wcs.WCS(header)
            c1 = SkyCoord(float(w.wcs_pix2world(0, 0, 0)[0])*u.degree, 
                          float(w.wcs_pix2world(0, 0, 0)[1])*u.degree, 
                          frame='icrs')
            c2 = SkyCoord(float(w.wcs_pix2world(img.shape[1], img.shape[0], 0)[0])*u.degree, 
                          float(w.wcs_pix2world(img.shape[1], img.shape[0], 0)[1])*u.degree, 
                          frame='icrs')
            c_cen = SkyCoord(float(w.wcs_pix2world(img.shape[1]//2, img.shape[0]//2, 0)[0])*u.degree, 
                             float(w.wcs_pix2world(img.shape[1]//2, img.shape[0]//2, 0)[1])*u.degree, 
                             frame='icrs')
            radius = c1.separation(c2).to(u.degree).value
            from astroquery.vizier import Vizier
            from astropy.coordinates import Angle
            Vizier.ROW_LIMIT = -1
            if method.lower() == 'apass':
                query_method = APASS_vizier
            elif method.lower() == 'usno':
                query_method = USNO_vizier
            else:
                raise ValueError("Method must be 'gaia', 'apass' or 'usno'!")
            result = Vizier.query_region(str(c_cen.ra.value) + ' ' + str(c_cen.dec.value), 
                                         radius=Angle(radius, "deg"), catalog=query_method)
            catalog = result.values()[0]
            catalog.rename_column('RAJ2000', 'ra')
            catalog.rename_column('DEJ2000', 'dec')
            if method.lower() == 'apass':
                catalog.rename_column('e_RAJ2000', 'e_ra')
                catalog.rename_column('e_DEJ2000', 'e_dec')
        return catalog
    else:
        raise ValueError("Method must be 'gaia', 'apass' or 'usno'!")
        return 

# Read a star catalog and translate (ra, dec) to (x, y)
def readStarCatalog(catalog, img, img_header, ra_dec_name=None, mag_name=None, bright_lim=15.5):
    """ Read a given catalog (astropy Table objects). Note that the column names of RA and DEC
        must be 'ra' and 'dec' or 'RA' and 'DEC'. One magnitude column shuold be included
        and named 'mag' or 'MAG'.

    Parameters:
        catalog: given catalog, must contain ra, dec and some magnitude (such as r-mag).
        ra_dec_name: ['ra', 'dec']
        img (2-D numpy array): image itselt.
        img_header: the header of this image.
        bright_lim: the magnitude limit of stars to be masked out. 

    Returns:
        xys: 2-D numpy table, contains 'x' and 'y' coordinate of bright stars.
        nx, ny: dimension of image.
    """
    if ra_dec_name is None:
        for item in catalog.colnames:
            if 'ra' in item.lower():
                ra_name = item
                break 
        for item in catalog.colnames:
            if 'de' in item.lower():
                dec_name = item
                break
    else:
        ra_name, dec_name = ra_dec_name

    if mag_name is None:
        cont = 0
        for item in catalog.colnames:
            if 'mag' in item.lower():
                mag_name = item
                cont += 1
        if cont > 1:
            for item in catalog.colnames:
                if 'm' in item.lower() and 'r' in item.lower():
                    mag_name = item
                    break
    
    ra = catalog[ra_name].data
    dec = catalog[dec_name].data
    mag = catalog[mag_name].data
    # convert ra,dec to x,y in the image
    w = wcs.WCS(img_header)
    x, y = w.wcs_world2pix(ra, dec, 0)
    # get dimensions of image
    # Here x is along RA direction, y is along DEC direction
    ny, nx = img.shape
    # iraf.center('_res.fits',coords='_coords_in.dat',output='_coords_out.dat',interact='no',
    #    wcsin='physical',wcsout='physical',calgori='gauss')
    x = (x).astype(np.int)
    y = (y).astype(np.int)

    '''
    # locations and magnitudes of stars that can be used to construct halo images.
    # 'b' represent 'stars that can be used to Build model'
    nh = 2.0
    xb = x[(mag < lim) & 
           (x > nh * halosize) & (x < nx - nh * halosize) &
           (y > nh * halosize) & (y < ny - nh * halosize)]
    yb = y[(mag < lim) & 
           (x > nh * halosize) & (x < nx - nh * halosize) &
           (y > nh * halosize) & (y < ny - nh * halosize)]
    magb = mag[(mag < lim) & 
               (x > nh * halosize) & (x < nx - nh * halosize) &
               (y > nh * halosize) & (y < ny - nh * halosize)]
    '''
    # locations and magnitudes of all stars in image
    xs = x[(mag < bright_lim) & (x > 0) & (x < nx) & (y > 0) & (y < ny)]
    ys = y[(mag < bright_lim) & (x > 0) & (x < nx) & (y > 0) & (y < ny)]
    mags = mag[(mag < bright_lim) & (x > 0) & (x < nx) & (y > 0) & (y < ny)]
    print('You have {} stars!'.format(len(xs)))
    xys = np.zeros(len(xs), dtype=[('x', int), ('y', int), ('mag', float)])
    xys['x'] = xs
    xys['y'] = ys
    xys['mag'] = mags
    #Table(xys).write('_coords_in.dat', format='ascii', overwrite=True)

    '''
    xyb = np.zeros(len(xb), dtype=[('x', int), ('y', int), ('mag', float)])
    xyb['x'] = xb
    xyb['y'] = yb
    xyb['mag'] = magb
    '''

    return (xys, nx, ny)

# Simply remove stars by masking them out
def mask_out_stars_hsc(segmap, img, header, method='gaia', bright_lim=15.5, catalog_dir=None):
    """ For HSC, the bleeding tails are annoying. So this function is for them.
    Mask out bright stars on the segmentation map of high resolution image, 
        before degrading to low resolution.

    Parameters:
        segmap (2-D numpy array): segmentation map, on which bright stars will be masked.
        img (2-D numpy array): image itselt.
        header: the header of this image.
        method (str): here three methods are provided: 'gaia', 'apass' or 'usno'.
        bright_lim (float): the magnitude limit of stars to be masked out. 
        catalog_dir (str): optional, you can provide local catalog here.

    Returns:
        segmap_cp: segmentation map after removing bright stars.
    """
    catalog = query_star(img, header, method=method, bright_lim=bright_lim)
    segmap_cp = copy.deepcopy(segmap)
    if method == 'gaia':
        xys, nx, ny = readStarCatalog(catalog, img, header, bright_lim=bright_lim, 
                            ra_dec_name=['ra', 'dec'], mag_name='phot_bp_mean_mag')
    else:
        xys, nx, ny = readStarCatalog(catalog, img, header, bright_lim=bright_lim)
    xs, ys = xys['x'], xys['y']
    for i in range(len(xys)):
        obj_id = segmap_cp[int(ys[i]), int(xs[i])]
        llim = obj_id - 0.1
        ulim = obj_id + 0.1
        segmap_cp[(segmap_cp < ulim) & (segmap_cp > llim)] = 0
        if (ys[i] + 5) < ny:
                obj_id = segmap_cp[int(ys[i]) + 5, int(xs[i])]
                llim = obj_id - 0.1
                ulim = obj_id + 0.1
                segmap_cp[(segmap_cp < ulim) & (segmap_cp > llim)] = 0
    return segmap_cp

def mask_out_stars(segmap, img, header, method='gaia', bright_lim=15.5, catalog_dir=None):
    """ Mask out bright stars on the segmentation map of high resolution image, 
        before degrading to low resolution.

    Parameters:
        segmap (2-D numpy array): segmentation map, on which bright stars will be masked.
        img (2-D numpy array): image itselt.
        header: the header of this image.
        method (str): here three methods are provided: 'gaia', 'apass' or 'usno'.
        bright_lim (float): the magnitude limit of stars to be masked out. 
        catalog_dir (str): optional, you can provide local catalog here.

    Returns:
        segmap_cp: segmentation map after removing bright stars.
    """
    catalog = query_star(img, header, method=method, bright_lim=bright_lim)
    segmap_cp = copy.deepcopy(segmap)
    if method == 'gaia':
        xys, nx, ny = readStarCatalog(catalog, img, header, bright_lim=bright_lim, 
                            ra_dec_name=['ra', 'dec'], mag_name='phot_bp_mean_mag')
    else:
        xys, nx, ny = readStarCatalog(catalog, img, header, bright_lim=bright_lim)
    xs, ys = xys['x'], xys['y']
    for i in range(len(xys)):
        obj_id = segmap_cp[int(ys[i]), int(xs[i])]
        llim = obj_id - 0.1
        ulim = obj_id + 0.1
        segmap_cp[(segmap_cp < ulim) & (segmap_cp > llim)] = 0
    return segmap_cp

# Mask certain galaxy
def mask_out_certain_galaxy(segmap, header, gal_cat=None):
    """ Mask out certain galaxy on segmentation map.

    Parameters:
        segmap (2-D numpy array): segmentation map, on which bright stars will be masked.
        img (2-D numpy array): image itselt.
        header: the header of this image.
        gal_cat (astropy.table.Table object): catalog of galaxies you want to mask out. 
            Must have columns called 'ra' and 'dec' (or 'x' and 'y').

    Returns:
        segmap_cp: segmentation map after removing bright stars.
    """
    from astropy import wcs
    if gal_cat is not None:
        w = wcs.WCS(header) # this header should be of the subsampled image
        segmap_cp = copy.copy(segmap)
        
        for item in gal_cat.colnames:
            if 'ra' in item.lower():
                coor_sys = 'sky'
                break
            elif 'x' in item.lower():
                coor_sys = 'image'
                break
            else:
                coor_sys = 'wrong'

        for obj in gal_cat:
            if coor_sys == 'sky':
                x, y = w.wcs_world2pix(obj['ra'], obj['dec'], 0)
            elif coor_sys == 'image':
                x, y = obj['x'] * 3, obj['y'] * 3
            elif coor_sys == 'wrong':
                raise ValueError("Wrong catalog format!")
            
            obj_id = np.int(segmap_cp[int(y), int(x)])
            print('### Removing object ' + str(obj_id) + ' from mask ###')
            llim = obj_id - 0.1
            ulim = obj_id + 0.1
            segmap_cp[(segmap_cp < ulim) & (segmap_cp > llim)] = 0
        return segmap_cp
    else:
        return segmap

def create_matching_kernel_custom(source_psf, target_psf, window=None):
    """
    Create a kernel to match 2D point spread functions (PSF) using the
    ratio of Fourier transforms.

    Parameters
    ----------
    source_psf : 2D `~numpy.ndarray`
        The source PSF.  The source PSF should have higher resolution
        (i.e. narrower) than the target PSF.  ``source_psf`` and
        ``target_psf`` must have the same shape and pixel scale.

    target_psf : 2D `~numpy.ndarray`
        The target PSF.  The target PSF should have lower resolution
        (i.e. broader) than the source PSF.  ``source_psf`` and
        ``target_psf`` must have the same shape and pixel scale.

    window : callable, optional
        The window (or taper) function or callable class instance used
        to remove high frequency noise from the PSF matching kernel.
        Some examples include:

        * `~photutils.psf.matching.HanningWindow`
        * `~photutils.psf.matching.TukeyWindow`
        * `~photutils.psf.matching.CosineBellWindow`
        * `~photutils.psf.matching.SplitCosineBellWindow`
        * `~photutils.psf.matching.TopHatWindow`

        For more information on window functions and example usage, see
        :ref:`psf_matching`.

    Returns
    -------
    kernel : 2D `~numpy.ndarray`
        The matching kernel to go from ``source_psf`` to ``target_psf``.
        The output matching kernel is normalized such that it sums to 1.
    """

    # inputs are copied so that they are not changed when normalizing
    source_psf = np.copy(np.asanyarray(source_psf))
    target_psf = np.copy(np.asanyarray(target_psf))

    if source_psf.shape != target_psf.shape:
        raise ValueError('source_psf and target_psf must have the same shape '
                         '(i.e. registered with the same pixel scale).')

    # ensure input PSFs are normalized
    #source_psf /= source_psf.sum()
    #target_psf /= target_psf.sum()

    source_otf = np.fft.fftshift(np.fft.fft2(source_psf))
    target_otf = np.fft.fftshift(np.fft.fft2(target_psf))
    ratio = target_otf / source_otf

    # apply a window function in frequency space
    if window is not None:
        ratio *= window(target_psf.shape)

    kernel = np.real(np.fft.fftshift((np.fft.ifft2(np.fft.ifftshift(ratio)))))
    return kernel

# new version of makekernel using `Celestial` object
def Makekernel(img_hires, img_lowres, obj, s, d, cval=np.nan, window=CosineBellWindow(alpha=1)):
    """ Given position, make kernel based on this position image. 
        To make 'makekernel' more general, all coordinates are local. 
        Here we don't normalize PSF! Don't use `photutils.psf.creat_matching_kernel`.

    Parameters:
        img_hires (2-D numpy array): high resolution image
        img_lowres (2-D numpy array): low resolution image
        x, y (float): position of star, where kernel is made
        s: kernel half-size, in pixel
        d: PSF half-size, in pixel

    Returns:
        kernel: kernel generated by given position
        hires_cut: the local cutout from high resolution image
        lowres_cut: the local cutout from low resolution image
    """
    from .celestial import Star
    # Don't use mask
    star_lowres = Star(img_lowres.image, img_lowres.header, obj, halosize=s)
    star_lowres.centralize()
    lowres_cut = copy.deepcopy(star_lowres.image)
    
    star_hires = Star(img_hires.image, img_hires.header, obj, halosize=s)
    star_hires.centralize()
    hires_cut = copy.deepcopy(star_hires.image)
    
    # create kernel
    # Here a window must be included to remove high frequency noise when doing FFT.
    kernel = create_matching_kernel_custom(hires_cut, lowres_cut, window=window)
    # Here don't normalize PSF! Don't use `photutils.psf.creat_matching_kernel`.
    
    # set outer regions to zero, but why??
    kernel_mask = np.zeros_like(kernel)

    x1, x2, y1, y2 = s - d, s + d + 1, s - d, s + d + 1
    kernel_mask[y1:y2, x1:x2] = 1
    
    kernel *= kernel_mask
    
    return kernel, hires_cut, lowres_cut

# new version of autokernel using `Celestial` object
def Autokernel(img_hires, img_lowres, s, d, object_cat_dir=None, 
              frac_maxflux=0.1, fwhm_upper=14, nkernels=20, border=50, cval=np.nan, show_figure=True):
    """ Automatically generate kernel from an image, 
        by detecting star position and make kernel from them.
        Here all images are subsampled, and all coordinates are in 
        subsampled image's coordinate.

    Parameters:
        img_hires (Celestial class): high resolution image
        img_lowres (Celestial class): low resolution image
        object_cat_dir (str): the directory of object catalog in `fits`.
            This catalog contains all stars that can be used to construct PSF of high-res image.
        s (float): kernel half-size, in pixel
        d (float): PSF half-size, in pixel
        frac_maxflux (float): only account for stars whose flux is 
            less than `frac_maxflux * maxflux`
        nkernels (int): number of stars, from which the kernel is generated
        border (int): don't select stars with in the border/edge of image
        
    Returns:
        kernel_median: the median kernel generated from given image
    """
    # Attempt to generate kernel automatically

    # read sextractor catalog
    if object_cat_dir is None:
        data = img_hires.image.byteswap().newbyteorder()
        obj_cat, _ = extract_obj(data, b=64, f=3, sigma=2.5, minarea=3, show_fig=False,
                                 deblend_nthresh=32, deblend_cont=0.0005, flux_aper=[3, 6])
    else:
        obj_cat = Table.read(object_cat_dir, format='fits')
    obj_cat.sort(keys='flux')
    obj_cat.reverse()

    # Select good stars
    x, y = obj_cat['x'].data, obj_cat['y'].data
    flux = obj_cat['flux'].data
    fwhm = obj_cat['fwhm_custom'].data
    ba = obj_cat['b'].data / obj_cat['a'].data
    # take median of 10 brightest objects as maximum - to remove "crazy" values if they exist
    maxflux = np.nanmedian(flux[:10])
    flux_lim = frac_maxflux * maxflux
    # get dimensions of image
    ny, nx = img_hires.image.shape
    # remove edges and objects above flux limit, and near the border, and shuold be very round and small.
    # This excludes those objects who 1) are not stars; 2) are saturated or exotic.
    non_edge_flag = np.logical_and.reduce([(flux > 0.01 * maxflux), (flux < flux_lim), (x > border),
                                            (x < nx - border), (y > border),
                                            (y < ny - border), (ba > 0.8)]) # (fwhm < 10)
    good_cat = obj_cat[non_edge_flag]
    good_cat.sort('flux')
    good_cat.reverse()     
    
    # create array for kernels, with odd dimensions
    kernels = np.zeros((nkernels, 2 * s + 1, 2 * s + 1))
    cuts_high = np.zeros((nkernels, 2 * s + 1, 2 * s + 1))
    cuts_low = np.zeros((nkernels, 2 * s + 1, 2 * s + 1))

    extreme_val = 1e9
    # take brightest `n` objects
    bad_indices = []
    for i, obj in enumerate(good_cat[:nkernels]):
        kernel, hires_cut, lowres_cut = Makekernel(img_hires, img_lowres, obj, s, d, cval=cval)
        cuts_low[i, :, :] = lowres_cut
        cuts_high[i, :, :] = hires_cut

        # discard this one if flux deviates too much.
        flux_measured = np.nansum(hires_cut)
        # test if measured flux is approximately expected flux
        deviation = (flux_measured - obj['flux']) / obj['flux']
        print('# Star {0}: flux deviation = {1:.3f}'.format(i, deviation))
        
        if -0.2 < deviation < 0.5:
            kernels[i, :, :] = kernel[:, :]
        else:
            kernels[i, :, :] = kernel[:, :] * 0 + extreme_val
            bad_indices.append(i)
            extreme_val = -1 * extreme_val
            print('\t Rejected ' + str(i))

    stack_set = np.delete(kernels, bad_indices, axis=0)
    print('You have {} good stars to generate the median kernel'.format(len(stack_set)))
    # median of kernels is final kernel
    kernel_median = np.nanmedian(stack_set, axis=0)
    #save_to_fits(kernel_median, 'kernel_median.fits')
    save_to_fits(cuts_low, '_all_low.fits', overwrite=True)
    save_to_fits(cuts_high, '_all_high.fits', overwrite=True)
    save_to_fits(kernels, '_all_kernels.fits', overwrite=True)

    if show_figure:
        fig, axes = plt.subplots(6, 8, figsize=(13, 10))
        data_set = [cuts_high, cuts_low, kernels]
        text_set = ['CFHT', 'DF', 'Kernel']
        for i in range(16):
            for j in range(3):
                ax = axes[j + 3 * (i // 8 - 1), i%8]
                if i == 0 or i == 8:
                    display_single(data_set[j][i], pixel_scale=2.5, 
                                   scale_bar=False, ax=ax, 
                                   add_text=text_set[j], text_fontsize=15)
                else:
                    display_single(data_set[j][i], pixel_scale=2.5, scale_bar=False, ax=ax)
                        
        plt.subplots_adjust(wspace=0.0, hspace=0.0)
        plt.savefig('./kernel_stars.png', bbox_inches='tight', dpi=150)

    return kernel_median, good_cat

# Generate a star mask, different from `mask_out_star` function
def bright_star_mask(mask, catalog, bright_lim=17.5, r=2.0):
    """ Mask out bright stars on the segmentation map of high resolution image, 
        before degrading to low resolution.

    Parameters:
        

    Returns:
        segmap_cp: segmentation map after removing bright stars.
    """
    import sep
    # Make stars to be zero on segmap
    for obj in catalog:
        if obj['mag'] < bright_lim and obj['b'] / obj['a'] > 0.6:
            sep.mask_ellipse(mask, obj['x'], obj['y'], obj['a'], obj['b'], obj['theta'], r=r)
    return mask

# Subtract background of PSF
def psf_bkgsub(psf, edge):
    """
    Subtract a background of PSF, estimated by the pixel value around the edges of PSF.

    Parameters:
        psf (numpy 2-D array): PSF itself
        edge (int): size of edge, within which we measure the local background
    """
    # subtract background of halomodel - made up of residuals of neighboring objects
    if edge < 3:
        edge = 3
    ny, nx = psf.shape
    uni = np.ones_like(psf)
    d = np.sum(psf) - np.sum(psf[edge: ny - edge, 
                                 edge: nx - edge])
    u = np.sum(uni) - np.sum(uni[edge: ny - edge, 
                                 edge: nx - edge])
    d /= u # Now 'd' is the background value per pixel
    psf_sub = psf - d
    return psf_sub


#########################################################################
########################## The Tractor related ##########################
#########################################################################

# Add sources to tractor
def add_tractor_sources(obj_cat, sources, w, shape_method='manual'):
    '''
    Add tractor sources to the sources list.

    Parameters:
    ----------
    obj_cat: astropy Table, objects catalogue.
    sources: list, to which we will add objects.
    w: wcs object.
    shape_method: string, 'manual' or 'decals'. If 'manual', it will adopt the 
                manually measured shapes. If 'decals', it will adopt 'DECaLS' 
                tractor shapes.

    Returns:
    --------
    sources: list of sources.
    '''
    from tractor import NullWCS, NullPhotoCal, ConstantSky
    from tractor.galaxy import GalaxyShape, DevGalaxy, ExpGalaxy, CompositeGalaxy
    from tractor.psf import Flux, PixPos, PointSource, PixelizedPSF, Image, Tractor
    from tractor.ellipses import EllipseE
    obj_type = np.array(list(map(lambda st: st.rstrip(' '), obj_cat['type'])))
    comp_galaxy = obj_cat[obj_type == 'COMP']
    dev_galaxy = obj_cat[obj_type == 'DEV']
    exp_galaxy = obj_cat[obj_type == 'EXP']
    rex_galaxy = obj_cat[obj_type == 'REX']
    psf_galaxy = obj_cat[obj_type =='PSF']

    if shape_method is 'manual':
        # Using manually measured shapes
        if sources is None:
            sources = []
        for obj in comp_galaxy:
            pos_x, pos_y = obj['x'], obj['y']
            sources.append(
                CompositeGalaxy(
                    PixPos(pos_x, pos_y), Flux(0.4 * obj['flux']),
                    GalaxyShape(obj['a_arcsec'] * 0.8, 0.9,
                                90.0 + obj['theta'] * 180.0 / np.pi),
                    Flux(0.6 * obj['flux']),
                    GalaxyShape(obj['a_arcsec'], obj['b_arcsec'] / obj['a_arcsec'],
                                90.0 + obj['theta'] * 180.0 / np.pi)))
        for obj in dev_galaxy:
            pos_x, pos_y = obj['x'], obj['y']
            sources.append(
                DevGalaxy(
                    PixPos(pos_x, pos_y), Flux(obj['flux']),
                    GalaxyShape(obj['a_arcsec'], (obj['b_arcsec'] / obj['a_arcsec']),
                                (90.0 + obj['theta'] * 180.0 / np.pi))))
        for obj in exp_galaxy:
            pos_x, pos_y = obj['x'], obj['y']
            sources.append(
                ExpGalaxy(
                    PixPos(pos_x, pos_y), Flux(obj['flux']),
                    GalaxyShape(obj['a_arcsec'], (obj['b_arcsec'] / obj['a_arcsec']),
                                (90.0 + obj['theta'] * 180.0 / np.pi))))
        for obj in rex_galaxy:
            pos_x, pos_y = obj['x'], obj['y']
            sources.append(
                ExpGalaxy(
                    PixPos(pos_x, pos_y), Flux(obj['flux']),
                    GalaxyShape(obj['a_arcsec'], (obj['b_arcsec'] / obj['a_arcsec']),
                                (90.0 + obj['theta'] * 180.0 / np.pi))))
        for obj in psf_galaxy:
            pos_x, pos_y = obj['x'], obj['y']
            sources.append(PointSource(PixPos(pos_x, pos_y), Flux(obj['flux'])))

        print("Now you have %d sources" % len(sources))

    elif shape_method is 'decals':
        ## Using DECaLS shapes
        if sources is None:
            sources = []
        for obj in comp_galaxy:
            pos_x, pos_y = obj['x'], obj['y']
            sources.append(
                CompositeGalaxy(
                    PixPos(pos_x, pos_y), Flux(0.4 * obj['flux']),
                    EllipseE(obj['shapeexp_r'], obj['shapeexp_e1'],
                             obj['shapeexp_e2']), Flux(0.6 * obj['flux']),
                    EllipseE(obj['shapedev_r'], obj['shapedev_e1'],
                             obj['shapedev_e2'])))
        for obj in dev_galaxy:
            pos_x, pos_y = obj['x'], obj['y']
            sources.append(
                DevGalaxy(
                    PixPos(pos_x, pos_y), Flux(obj['flux']),
                    EllipseE(obj['shapedev_r'], obj['shapedev_e1'],
                             -obj['shapedev_e2'])))
        for obj in exp_galaxy:
            pos_x, pos_y = obj['x'], obj['y']
            sources.append(
                ExpGalaxy(
                    PixPos(pos_x, pos_y), Flux(obj['flux']),
                    EllipseE(obj['shapeexp_r'], obj['shapeexp_e1'],
                             -obj['shapeexp_e2'])))
        for obj in rex_galaxy:
            #if obj['point_source'] > 0.0:
            #            sources.append(PointSource(PixPos(w.wcs_world2pix([[obj['ra'], obj['dec']]],1)[0]),
            #                                               Flux(obj['flux'])))
            pos_x, pos_y = obj['x'], obj['y']
            sources.append(
                ExpGalaxy(
                    PixPos(pos_x, pos_y), Flux(obj['flux']),
                    EllipseE(obj['shapeexp_r'], obj['shapeexp_e1'],
                             -obj['shapeexp_e2'])))

        for obj in psf_galaxy:
            pos_x, pos_y = obj['x'], obj['y']
            sources.append(PointSource(PixPos(pos_x, pos_y), Flux(obj['flux'])))

        print("Now you have %d sources" % len(sources))
    else:
         raise ValueError('Cannot use this shape method') 
    return sources

# Do tractor iteration
def tractor_iteration(obj_cat, w, img_data, invvar, psf_obj, pixel_scale, shape_method='manual', 
                      kfold=4, first_num=50, fig_name=None):
    '''
    Run tractor iteratively.

    Parameters:
    -----------
    obj_cat: objects catalogue.
    w: wcs object.
    img_data: 2-D np.array, image.
    invvar: 2-D np.array, inverse variance matrix of the image.
    psf_obj: PSF object, defined by tractor.psf.PixelizedPSF() class.
    pixel_scale: float, pixel scale in unit arcsec/pixel.
    shape_method: if 'manual', then adopt manually measured shape. If 'decals', then adopt DECaLS shape from tractor files.
    kfold: int, iteration time.
    first_num: how many objects will be fit in the first run.
    fig_name: string, if not None, it will save the tractor subtracted image to the given path.

    Returns:
    -----------
    sources: list, containing tractor model sources.
    trac_obj: optimized tractor object after many iterations.
    '''
    from tractor import NullWCS, NullPhotoCal, ConstantSky
    from tractor.galaxy import GalaxyShape, DevGalaxy, ExpGalaxy, CompositeGalaxy
    from tractor.psf import Flux, PixPos, PointSource, PixelizedPSF, Image, Tractor
    from tractor.ellipses import EllipseE

    step = int((len(obj_cat) - first_num)/(kfold-1))
    for i in range(kfold):
        if i == 0:
            obj_small_cat = obj_cat[:first_num]
            sources = add_tractor_sources(obj_small_cat, None, w, shape_method='manual')
        else:
            obj_small_cat = obj_cat[first_num + step*(i-1) : first_num + step*(i)]
            sources = add_tractor_sources(obj_small_cat, sources, w, shape_method='manual')

        tim = Image(data=img_data,
                    invvar=invvar,
                    psf=psf_obj,
                    wcs=NullWCS(pixscale=pixel_scale),
                    sky=ConstantSky(0.0),
                    photocal=NullPhotoCal()
                    )
        trac_obj = Tractor([tim], sources)
        trac_mod = trac_obj.getModelImage(0, minsb=0.0)

        # Optimization
        trac_obj.freezeParam('images')
        trac_obj.optimize_loop()
        ########################
        plt.rc('font', size=20)
        if i % 2 == 1 or i == (kfold-1) :
            fig, [ax1, ax2, ax3] = plt.subplots(1, 3, figsize=(18,8))

            trac_mod_opt = trac_obj.getModelImage(0, minsb=0., srcs=sources[:])

            ax1 = display_single(img_data, ax=ax1, scale_bar=False)
            ax1.set_title('raw image')
            ax2 = display_single(trac_mod_opt, ax=ax2, scale_bar=False, contrast=0.02)
            ax2.set_title('tractor model')
            ax3 = display_single(abs(img_data - trac_mod_opt), ax=ax3, scale_bar=False, color_bar=True, contrast=0.05)
            ax3.set_title('residual')

            if i == (kfold-1):
                if fig_name is not None:
                    plt.savefig(fig_name, dpi=200, bbox_inches='tight')
                    plt.show(block=False)
                    print('The chi-square is', np.sqrt(np.mean(np.square((img_data - trac_mod_opt).flatten()))))
            else:
                plt.show()
                print('The chi-square is', np.sqrt(np.mean(np.square((img_data - trac_mod_opt).flatten()))) / np.sum(img_data)) 

        #trac_mod_opt = trac_obj.getModelImage(0, minsb=0., srcs=sources[1:])
        #ax4 = display_single(img_data - trac_mod_opt, ax=ax4, scale_bar=False, color_bar=True, contrast=0.05)
        #ax4.set_title('remain central galaxy')


    return sources, trac_obj, fig


#########################################################################
########################## YAML related #################################
#########################################################################

class Config(object):
    def __init__(self, d):
        for a, b in d.items():
            if isinstance(b, (list, tuple)):
                setattr(self, a, [Config(x) if isinstance(x, dict) else x for x in b])
            else:
                setattr(self, a, Config(b) if isinstance(b, dict) else b)

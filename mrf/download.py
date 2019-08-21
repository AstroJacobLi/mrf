import os
import copy
import numpy as np
from astropy import wcs
from astropy.io import fits
from astropy.table import Table, Column
from tqdm import tqdm

__all__ = ["TqdmUpTo", "megapipe_query_sql", "get_megapipe_catalog", "overlap_fraction",
    "wget_cfht", "download_cfht_megapipe", "download_highres"]

class TqdmUpTo(tqdm):
    """
    Provides ``update_to(n)`` which uses ``tqdm.update(delta_n)``.
    """

    def update_to(self, b=1, bsize=1, tsize=None):
        """
        b (int, optional): Number of blocks transferred so far [default: 1].
        bsize (int, optional): Size of each block (in tqdm units) [default: 1].
        tsize (int, optional): Total size (in tqdm units). If [default: None] remains unchanged.

        """
        if tsize is not None:
            self.total = tsize
        self.update(b * bsize - self.n)  # will also set self.n = b * bsize

def megapipe_query_sql(ra, dec, size):
    """ 
    Return SQL query command of CFHT megapipe

    Parameters:
        ra (float): in degree
        dec (float): in degree
        size (float): in degree

    Returns:
        url (str): The query URL, need to be opened by `wget` or `curl`.
    """
    
    return ("https://www.cadc-ccda.hia-iha.nrc-cnrc.gc.ca/tap/sync?REQUEST=doQuery&LANG=ADQL&FORMAT=CSV&QUERY="
            "SELECT * "
            "FROM caom2.Observation AS o JOIN caom2.Plane AS p ON o.obsID=p.obsID "
            "WHERE INTERSECTS(p.position_bounds, CIRCLE('ICRS', " + str(ra) + ', ' + str(dec) + ', ' + str(size) + ')) = 1 '
            "AND p.calibrationLevel >= 1 AND o.collection='CFHTMEGAPIPE' ")
            #"AND o.observationID LIKE 'MegaPipe%'")

def get_megapipe_catalog(ra, dec, size, output_filename='_megapipe_cat.csv', overwrite=True):
    """ 
    Download CFHT MegaPipe frame catalog of given ``(ra, dec)`` and image size, and save in ``csv`` format.

    Parameters:
        ra (float): in degree
        dec (float): in degree
        size: use ``astropy.units``. e.g. ``size = 300 * u.arcsec``.
            If not in `astropy.units` form, then default units are 'arcsec'.
        output_filename (string): The file will be saved here.
        overwrite (bool): it will overwrite the catalog if `overwrite=True`.

    Returns:
        None
    """

    import astropy.units as u
    if str(size).replace('.', '', 1).isdigit():
        size = size * u.arcsec
    size_deg = size.to(u.degree).value
    print('# Retrieving CFHT MegaPipe catalog!')
    sql = megapipe_query_sql(ra, dec, size_deg)
    if os.path.isfile(output_filename):
        if overwrite:
            os.remove(output_filename)
            os.system("wget -O " + output_filename + ' ' + '"' + sql + '"')
            print('# CFHT MegaPipe catalog retrieved successfully! It is saved as ' + output_filename)
        else:
            raise FileExistsError('!!!The catalog "' + output_filename + '" already exists!!!')
    else:
        os.system("wget -O " + output_filename + ' ' + '"' + sql + '"')
        print('# CFHT MegaPipe catalog retrieved successfully! It is saved as ' + output_filename)

def overlap_fraction(img, header, frame, verbose=True):
    """
    Calculate the overlap between a given CFHT frame and a given image. 

    Parameters:
        img (numpy 2-D array): image array.
        header: header of the image, typically ``astropy.io.fits.header`` object.
        frame: This is one row of ``mega_cat``, containing "position_bounds".
        verbose (bool): Whether print out intersection percentage.

    Return:
        percentage (float): The overlapping fraction (with respect to the input CFHT frame).
    """
    # Frame is one row of mega_cat.
    from shapely.geometry import Polygon
    w = wcs.WCS(header)
    ra_bound = list(map(float, frame['position_bounds'].split(' ')))[0::2]
    dec_bound = list(map(float, frame['position_bounds'].split(' ')))[1::2]
    x_bound, y_bound = w.wcs_world2pix(ra_bound, dec_bound, 0)
    img_bound = Polygon(list(zip([0, img.shape[1], img.shape[1], 0], 
                                    [0, 0, img.shape[0], img.shape[0]])))
    frame_bound = Polygon(list(zip(x_bound, y_bound)))
    if frame_bound.contains(img_bound):
        if verbose:
            print('# The intersection accounts for 100 %')
        return 1
    elif not frame_bound.overlaps(img_bound):
        print('# No Intersection!')
        return 0
    else:
        intersect = frame_bound.intersection(img_bound)
        if verbose:
            print('# The intersection accounts for {} %'.format(
                round(intersect.area / img_bound.area * 100, 2)))
        percentage = intersect.area / img_bound.area
        return percentage

def wget_cfht(frame, band, output_dir, output_name, overwrite=True):
    ### Download frame ###
    #from compsub.utils import TqdmUpTo
    import urllib
    print('# Downloading Frame ... Please Wait!!!')
    URL = 'https://www.cadc-ccda.hia-iha.nrc-cnrc.gc.ca/data/pub/CFHTSG/' + frame[
            'productID'] + '.fits'
    filename = output_name + '_' + band + '.fits'
    if not os.path.isfile(filename):
        with TqdmUpTo(unit='B', unit_scale=True, miniters=1, desc=frame['productID']) as t:  # all optional kwargs
            urllib.request.urlretrieve(URL, filename=output_dir + filename,
                                    reporthook=t.update_to, data=None)
        print('# Downloading ' + filename + ' finished! ') 
    elif os.path.isfile(filename) and overwrite:
        os.remove(filename)
        with TqdmUpTo(unit='B', unit_scale=True, miniters=1, desc=frame['productID']) as t:  # all optional kwargs
            urllib.request.urlretrieve(URL, filename=output_dir + filename,
                                    reporthook=t.update_to, data=None)
        print('# Downloading ' + filename + ' finished! ')                            
    elif os.path.isfile(filename) and not overwrite:
        print('!!!The image "' + output_dir + filename + '" already exists!!!')
    return

def download_cfht_megapipe(img, header, band='g', mega_cat_dir='_megapipe_cat.csv', 
                           output_dir='./', output_name='CFHT_megapipe_img', overwrite=True):
    """ 
    Download CFHT MegaPipe frame catalog of given (ra, dec) and image size. 
    This could be really **TIME CONSUMING**!!! Typically one frame could be 500M to 2G.

    Parameters:
        img (numpy 2-D array): input image, usually low-resolution image (such as Dragonfly).
        header (astropy.io.fits.header object): header of the input image.
        band (string): such as 'g' ('G') and 'r' ('R').
        mega_cat_dir (string): the directory of MegaPipe catalog, 
            which is downloaded by `get_megapipe_catalog`.
        output_dir (string): the CFHT images will be downloaded here.
        output_name (string): name of CFHT images, as you can change
        overwrite (bool): It will overwrite the image if `overwrite=True`.

    Returns:
        None
    """

    print('# Removing frames with small overlaps from the MegaPipe catalog ...')
    if not os.path.isfile(mega_cat_dir):
        raise ValueError('Cannot find MegaPipe catalog at: "' + mega_cat_dir + '"!')
    else:
        mega_cat = Table.read(mega_cat_dir, format='pandas.csv')

    print('# Selecting frames with given filter ...')
    mega_cat = mega_cat[[band.lower() in item for item in mega_cat['energy_bandpassName'].data]]

    w = wcs.WCS(header)
    # Listen, in w.wcs_pix2world(), it's (x, y) on the image (x along horizontal/RA, y along vertical/DEC)!!!!
    ra_cen, dec_cen = w.wcs_pix2world(img.shape[1]//2, img.shape[0]//2, 0)
    ra_corner, dec_corner = w.wcs_pix2world([0, img.shape[1], img.shape[1], 0], 
                                            [0, 0, img.shape[0], img.shape[0]], 0)
        
    # Calculate the overlap between the given image and each CFHT frame
    overlap = [overlap_fraction(img, header, frame, is_print=False) for frame in mega_cat]
    mega_cat.add_column(Column(data = np.array(overlap), name='overlap'))
    mega_cat.sort('overlap')
    mega_cat.reverse()
    overlap = mega_cat['overlap'].data
    mega_cat.write(mega_cat_dir, format='pandas.csv')

    if np.amax(overlap) >= 0.6:
        mega_cat = mega_cat[overlap > 0.6]
        status = 'good'
    else:
        mega_cat = mega_cat[(overlap >= 0.3)]
        status = 'bad'
    

    # Download multiple images
    if any(['MegaPipe' in obj['productID'] for obj in mega_cat]):
        flag = ['MegaPipe' in obj['productID'] for obj in mega_cat]
        frame_list = mega_cat[flag]
        # MegaPipe.xxx image doesn't have an exposure time
        print('# The frame to be downloaded is ', frame_list['productID'].data)
        for i, frame in enumerate(frame_list):
            wget_cfht(frame, band=band, output_dir=output_dir, 
                      output_name=output_name + '_' +str(i), overwrite=overwrite)
    else: # Select those with longest exposure time
        print('# Choosing the frame with longest exposure: ' + str(np.amax(mega_cat['time_exposure'])) + 's!')
        frame = mega_cat[np.argmax(mega_cat['time_exposure'])]
        print('# The frame to be downloaded is ' + frame['productID'])
        wget_cfht(frame, band=band, output_dir=output_dir, 
                  output_name=output_name, overwrite=overwrite)
    
def download_highres(lowres_dir, high_res='hsc', band='g', overwrite=False):
    """ Download high resolution image which overlaps with the given low resolution image.
        This could be **TIME CONSUMING**! Typically one frame could be 500M to 2G.
        You also need to install ``unagi`` (https://github.com/dr-guangtou/unagi) to download HSC images. 
        
    Parameters:
        lowres_dir (string): the directory of input low-resolution image.
        high_res (string): name of high-resolution survey, now support 'HSC' and 'CFHT'.
        band (string): name of filter, typically 'g' or 'r'.
        overwrite (bool): it will overwrite the image if `overwrite=True`.

    Returns:
        None
    """

    from astropy.coordinates import SkyCoord
    import astropy.units as u
    from astropy import wcs
    hdu = fits.open(lowres_dir)
    img = hdu[0].data
    header = hdu[0].header
    w = wcs.WCS(header)
    hdu.close()
    # Calculate the (diagnal) size of image in degrees
    c1 = SkyCoord(float(w.wcs_pix2world(0, 0, 0)[0]), 
                  float(w.wcs_pix2world(0, 0, 0)[1]), 
                  frame='icrs', unit='deg')
    c2 = SkyCoord(float(w.wcs_pix2world(img.shape[1], img.shape[0], 0)[0]), 
                  float(w.wcs_pix2world(img.shape[1], img.shape[0], 0)[1]), 
                  frame='icrs', unit='deg')
    c_cen = SkyCoord(float(w.wcs_pix2world(img.shape[1]//2, img.shape[0]//2, 0)[0]), 
                     float(w.wcs_pix2world(img.shape[1]//2, img.shape[0]//2, 0)[1]), 
                     frame='icrs', unit='deg')   
    print('# The center of input image is: ', c_cen.ra, c_cen.dec)
    radius = c1.separation(c2).to(u.degree)
    print('# The diagnal size of input low-resolution image is ' + str(radius))
    
    if high_res.lower() == 'hsc': 
        if radius / 1.414 > 2116 * u.arcsec:
            raise ValueError('# Input image size is too large for HSC! Try other methods!')
        try:
            from unagi import hsc
            from unagi.task import hsc_cutout, hsc_tricolor, hsc_check_coverage
        except:
            raise ImportError('You should install `unagi` https://github.com/dr-guangtou/unagi to download HSC images! ')
        # Setup HSC server
        pdr2 = hsc.Hsc(dr='pdr2', rerun='pdr2_wide')
        #pdr2 = hsc.Hsc(dr='dr2', rerun='s18a_wide')
        # Check if it is within the footprint
        cover = hsc_check_coverage(c_cen, archive=pdr2, verbose=False, return_filter=True)
        if len(cover) == 0:
            raise ValueError('# This region is NOT in the footprint of HSC PDR2!')
        # Angular size (must with unit!)
        else:
            if os.path.isfile('hsc_cutout_' + band):
                if not overwrite:
                    raise FileExistsError('# hsc_cutout_' + band + 'already exists!')
            else:
                cutout = hsc_cutout(c_cen,
                                    cutout_size=radius / np.sqrt(2) / 2,
                                    filters=band,
                                    archive=pdr2,
                                    use_saved=False,
                                    mask=False,
                                    variance=False,
                                    output_dir='./',
                                    prefix='hsc_cutout',
                                    verbose=True,
                                    save_output=True)
                cutout.close()
    elif high_res.lower() == 'cfht':
        get_megapipe_catalog(c_cen.ra.value, c_cen.dec.value, radius / 2)
        download_cfht_megapipe(img, header, band=band.upper(), output_dir='./', overwrite=overwrite)
    else:
        raise ValueError('# This survey is not supported yet. Please use "HSC" or "CFHT"!')
    return


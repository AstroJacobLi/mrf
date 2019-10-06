import os
import copy
import numpy as np
from astropy import wcs
from astropy.io import fits
import astropy.units as u
from astropy.table import Table, Column
from tqdm import tqdm
from . import DECaLS_pixel_scale

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
    overlap = [overlap_fraction(img, header, frame, verbose=False) for frame in mega_cat]
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

def download_decals_cutout(ra, dec, size, band, layer='dr8-south', pixel_unit=False, 
                    output_dir='./', output_name='DECaLS_img', overwrite=True):
    '''Download DECaLS small image cutout of a given image. Maximum size is 3000 * 3000 pix.
    
    Parameters:
        ra (float): RA (degrees)
        dec (float): DEC (degrees)
        size (float): image size in pixel or arcsec. If pixel_unit = True, it's in pixel.
        band (string): such as 'r' or 'g'
        layer (string): data release of DECaLS. If your object is too north, try 'dr8-north'. 
            For details, please check http://legacysurvey.org/dr8/description/.
        pixel_unit (bool): If true, size will be in pixel unit.
        output_dir (str): directory of output files.
        output_name (str): prefix of output images.
        overwrite (bool): overwrite files or not.

    Return:
        None
    '''
    import urllib

    if pixel_unit is False:
        s = size / DECaLS_pixel_scale
    else:
        s = size
    
    URL = 'http://legacysurvey.org/viewer/fits-cutout?ra={0}&dec={1}&pixscale={2}&layer={3}&size={4:.0f}&bands={5}'.format(ra, dec, DECaLS_pixel_scale, layer, s, band)
    
    filename = output_name + '_' + band + '.fits'
    if not os.path.isfile(filename):
        with TqdmUpTo(unit='B', unit_scale=True, miniters=1, desc=filename) as t:  # all optional kwargs
            urllib.request.urlretrieve(URL, filename=output_dir + filename,
                                    reporthook=t.update_to, data=None)
        print('# Downloading ' + filename + ' finished! ') 
    elif os.path.isfile(filename) and overwrite:
        os.remove(filename)
        with TqdmUpTo(unit='B', unit_scale=True, miniters=1, desc=filename) as t:  # all optional kwargs
            urllib.request.urlretrieve(URL, filename=output_dir + filename,
                                    reporthook=t.update_to, data=None)
        print('# Downloading ' + filename + ' finished! ')                            
    elif os.path.isfile(filename) and not overwrite:
        print('!!!The image "' + output_dir + filename + '" already exists!!!')
    return
    
def download_decals_brick(brickname, band, layer='dr8-south', output_dir='./', 
                          output_name='DECaLS', overwrite=True, verbose=True):
    '''Generate URL of the DECaLS coadd of a single brick.
    
    Parameters:
        brickname (string): the name of the brick, such as "0283m005".
        band (string): such as 'r' or 'g'.
        layer (string): data release of DECaLS. If your object is too north, try 'dr8-north'. 
            For details, please check http://legacysurvey.org/dr8/description/.
        output_dir (str): directory of output files.
        output_name (str): prefix of output images.
        overwrite (bool): overwrite files or not.

    Return:
        None
    '''
    import urllib

    if layer == 'dr8-north':
        URL = 'http://portal.nersc.gov/project/cosmo/data/legacysurvey/dr8/north/coadd/{0}/{1}/legacysurvey-{2}-image-{3}.fits.fz'.format(brickname[:3], brickname, brickname, band)
    else:
        URL = 'http://portal.nersc.gov/project/cosmo/data/legacysurvey/dr8/south/coadd/{0}/{1}/legacysurvey-{2}-image-{3}.fits.fz'.format(brickname[:3], brickname, brickname, band)

    filename = output_name + '_' + brickname + '_' +  band + '.fits'

    if not os.path.isfile(filename):
        if verbose:
            with TqdmUpTo(unit='B', unit_scale=True, miniters=1, desc=filename) as t:  # all optional kwargs
                urllib.request.urlretrieve(URL, filename=output_dir + filename,
                                        reporthook=t.update_to, data=None)
        else:
            urllib.request.urlretrieve(URL, filename=output_dir + filename, data=None)
        print('# Downloading ' + filename + ' finished! ') 

    elif os.path.isfile(filename) and overwrite:
        os.remove(filename)
        if verbose:
            with TqdmUpTo(unit='B', unit_scale=True, miniters=1, desc=filename) as t:  # all optional kwargs
                urllib.request.urlretrieve(URL, filename=output_dir + filename,
                                        reporthook=t.update_to, data=None)
        else:
            urllib.request.urlretrieve(URL, filename=output_dir + filename, data=None)
        print('# Downloading ' + filename + ' finished! ') 

    elif os.path.isfile(filename) and not overwrite:
        print('!!!The image "' + output_dir + filename + '" already exists!!!')

    return

def download_decals_large(ra, dec, band, size=0.7*u.deg, radius=0.5*u.deg, layer='dr8-south', verbose=True,
                          output_dir='./', output_name='DECaLS', overwrite=True):
    '''Download bricks and stitch them together using ``swarp``. Hence ``swarp`` must be installed!
    ``swarp`` resamples the image, but doesn't correct background.

    Parameters:
        ra (float): RA of the object.
        dec (float): DEC of the object.
        band: string, such as 'r' or 'g'.
        size (``astropy.units`` object): size of cutout, it should be comparable to ``radius``.
        radius (``astropy.units`` object): bricks whose distances to the object are 
            nearer than this radius will be download.  
        layer (str): data release of DECaLS. If your object is too north, try 'dr8-north'. 
            For details, please check http://legacysurvey.org/dr8/description/.
        output_dir (str): directory of output files.
        output_name (str): prefix of output images.
        overwrite (bool): overwrite files or not.
    
    Return:
        None
    '''

    import urllib
    from astropy.coordinates import SkyCoord
    from .utils import save_to_fits
    import subprocess

    ## Download survey-brick.fits
    URL = 'https://portal.nersc.gov/project/cosmo/data/legacysurvey/dr8/survey-bricks.fits.gz'

    if os.path.isfile('_survey_brick.fits'):
        os.remove('_survey_brick.fits')
    urllib.request.urlretrieve(URL, filename='_survey_brick.fits', data=None)

    # Find nearby bricks 
    bricks_cat = Table.read('_survey_brick.fits', format='fits')
    bricks_sky = SkyCoord(ra=np.array(bricks_cat['RA']), 
                          dec=np.array(bricks_cat['DEC']), unit='deg')
    object_coord = SkyCoord(ra, dec, unit='deg')
    to_download = bricks_cat[bricks_sky.separation(object_coord) <= radius]
    print('# You have {} bricks to be downloaded.'.format(len(to_download)))
    filenameset = []
    for obj in to_download:
        file = '_brick_' + obj['BRICKNAME'] + '_{}.fits'.format(band)
        filenameset.append(os.path.join(output_dir, file))
        if not os.path.isfile(file):
            download_decals_brick(obj['BRICKNAME'], band.lower(), output_name='_brick', 
                                output_dir=output_dir, verbose=verbose)
            hdu = fits.open(os.path.join(output_dir, file))
            img = hdu[1].data
            hdr = hdu[1].header
            hdr['XTENSION'] = 'IMAGE'
            hdu.close()
            save_to_fits(img, os.path.join(output_dir, file), header=hdr);

    # Calculating image size in pixels
    imgsize = int(size.to(u.arcsec).value / DECaLS_pixel_scale)
    # Configure ``swarp``
    with open("config_swarp.sh","w+") as f:
        # check if swarp is installed
        f.write('for cmd in swarp; do\n')
        f.write('\t hasCmd=$(which ${cmd} 2>/dev/null)\n')
        f.write('\t if [[ -z "${hasCmd}" ]]; then\n')
        f.write('\t\t echo "This script requires ${cmd}, which is not in your \$PATH." \n')
        f.write('\t\t exit 1 \n')
        f.write('\t fi \n done \n\n')
        
        # Write ``default.swarp``.
        f.write('/bin/rm -f default.swarp \n')
        f.write('cat > default.swarp <<EOT \n')
        f.write('IMAGEOUT_NAME \t\t {}.fits      # Output filename\n'.format(os.path.join(output_dir, '_'.join([output_name, band]))))
        f.write('WEIGHTOUT_NAME \t\t {}_weights.fits     # Output weight-map filename\n\n'.format(os.path.join(output_dir, '_'.join([output_name, band]))))
        f.write('HEADER_ONLY            N               # Only a header as an output file (Y/N)?\nHEADER_SUFFIX          .head           # Filename extension for additional headers\n\n')
        f.write('#------------------------------- Input Weights --------------------------------\n\nWEIGHT_TYPE            NONE            # BACKGROUND,MAP_RMS,MAP_VARIANCE\n                                       # or MAP_WEIGHT\nWEIGHT_SUFFIX          weight.fits     # Suffix to use for weight-maps\nWEIGHT_IMAGE                           # Weightmap filename if suffix not used\n                                       # (all or for each weight-map)\n\n')
        f.write('#------------------------------- Co-addition ----------------------------------\n\nCOMBINE                Y               # Combine resampled images (Y/N)?\nCOMBINE_TYPE           MEDIAN          # MEDIAN,AVERAGE,MIN,MAX,WEIGHTED,CHI2\n                                       # or SUM\n\n')
        f.write('#-------------------------------- Astrometry ----------------------------------\n\nCELESTIAL_TYPE         NATIVE          # NATIVE, PIXEL, EQUATORIAL,\n                                       # GALACTIC,ECLIPTIC, or SUPERGALACTIC\nPROJECTION_TYPE        TAN             # Any WCS projection code or NONE\nPROJECTION_ERR         0.001           # Maximum projection error (in output\n                                       # pixels), or 0 for no approximation\nCENTER_TYPE            MANUAL          # MANUAL, ALL or MOST\n')
        f.write('CENTER   {0}, {1} # Image Center\n'.format(ra, dec))
        f.write('PIXELSCALE_TYPE        MANUAL          # MANUAL,FIT,MIN,MAX or MEDIAN\n')
        f.write('PIXEL_SCALE            {}  # Pixel scale\n'.format(DECaLS_pixel_scale))
        f.write('IMAGE_SIZE             {0},{1} # scale = 0.262 arcsec/pixel\n\n'.format(imgsize, imgsize))
        f.write('#-------------------------------- Resampling ----------------------------------\n\nRESAMPLE               Y               # Resample input images (Y/N)?\nRESAMPLE_DIR           .               # Directory path for resampled images\nRESAMPLE_SUFFIX        .resamp.fits    # filename extension for resampled images\n\nRESAMPLING_TYPE        LANCZOS3        # NEAREST,BILINEAR,LANCZOS2,LANCZOS3\n                                       # or LANCZOS4 (1 per axis)\nOVERSAMPLING           0               # Oversampling in each dimension\n                                       # (0 = automatic)\nINTERPOLATE            N               # Interpolate bad input pixels (Y/N)?\n                                       # (all or for each image)\n\nFSCALASTRO_TYPE        FIXED           # NONE,FIXED, or VARIABLE\nFSCALE_KEYWORD         FLXSCALE        # FITS keyword for the multiplicative\n                                       # factor applied to each input image\nFSCALE_DEFAULT         1.0             # Default FSCALE value if not in header\n\nGAIN_KEYWORD           GAIN            # FITS keyword for effect. gain (e-/ADU)\nGAIN_DEFAULT           0.0             # Default gain if no FITS keyword found\n\n')
        f.write('#--------------------------- Background subtraction ---------------------------\n\nSUBTRACT_BACK          N               # Subtraction sky background (Y/N)?\n                                       # (all or for each image)\n\nBACK_TYPE              AUTO            # AUTO or MANUAL\n                                       # (all or for each image)\nBACK_DEFAULT           0.0             # Default background value in MANUAL\n                                       # (all or for each image)\nBACK_SIZE              128             # Background mesh size (pixels)\n                                       # (all or for each image)\nBACK_FILTERSIZE        3               # Background map filter range (meshes)\n                                       # (all or for each image)\n\n')
        f.write('#------------------------------ Memory management -----------------------------\n\nVMEM_DIR               .               # Directory path for swap files\nVMEM_MAX               2047            # Maximum amount of virtual memory (MB)\nMEM_MAX                2048            # Maximum amount of usable RAM (MB)\nCOMBINE_BUFSIZE        1024            # Buffer size for combine (MB)\n\n')
        f.write('#------------------------------ Miscellaneous ---------------------------------\n\nDELETE_TMPFILES        Y               # Delete temporary resampled FITS files\n                                       # (Y/N)?\nCOPY_KEYWORDS          OBJECT          # List of FITS keywords to propagate\n                                       # from the input to the output headers\nWRITE_FILEINFO         Y               # Write information about each input\n                                       # file in the output image header?\nWRITE_XML              N               # Write XML file (Y/N)?\nXML_NAME               swarp.xml       # Filename for XML output\nVERBOSE_TYPE           QUIET           # QUIET,NORMAL or FULL\n\nNTHREADS               0               # Number of simultaneous threads for\n                                       # the SMP version of SWarp\n                                       # 0 = automatic \n')
        f.write('EOT\n')
        f.write('swarp ' + ' '.join(filenameset) + '\n\n')
        f.write('rm ' + os.path.join(output_dir, '_*'))
        f.close()
    
    os.system('/bin/bash config_swarp.sh')
    print('# The image is save as {}'.format(os.path.join(output_dir, '_'.join([output_name, band]))))

def download_hsc_large(ra, dec, band, size=0.7*u.deg, radius=0.5*u.deg, verbose=True,
                       output_dir='./', output_name='HSC_large', overwrite=True):
    '''Download HSC patches and stitch them together using ``swarp``. Hence ``swarp`` must be installed!
    ``swarp`` resamples the image, but doesn't correct background.

    Parameters:
        ra (float): RA of the object.
        dec (float): DEC of the object.
        band: string, such as 'r' or 'g'.
        size (``astropy.units`` object): size of cutout, it should be comparable to ``radius``.
        radius (``astropy.units`` object): bricks whose distances to the object are 
            nearer than this radius will be download.  
        output_dir (str): directory of output files.
        output_name (str): prefix of output images.
        overwrite (bool): overwrite files or not.
    
    Return:
        None
    '''

    import urllib
    from astropy.coordinates import SkyCoord
    from .utils import save_to_fits
    import subprocess

    ## Setup ``unagi``
    try:
        from unagi import hsc
    except:
        raise ImportError('`unagi` (https://github.com/dr-guangtou/unagi) must be installed to download HSC data!')
    pdr2 = hsc.Hsc(dr='pdr2', rerun='pdr2_wide')

    ## Download survey-summary
    URL = 'https://github.com/AstroJacobLi/slug/raw/master/demo/HSC_tracts_patches_pdr2_wide.fits'

    if os.path.isfile('_survey_summary.fits'):
        os.remove('_survey_summary.fits')
    urllib.request.urlretrieve(URL, filename='_survey_summary.fits', data=None)

    # Find nearby bricks 
    patch_cat = Table.read('_survey_summary.fits', format='fits')
    patch_sky = SkyCoord(ra=np.array(patch_cat['ra_cen']), 
                        dec=np.array(patch_cat['dec_cen']), unit='deg')
    object_coord = SkyCoord(ra, dec, unit='deg')
    flag = patch_sky.separation(object_coord) <= radius
    to_download = patch_cat[flag]
    distance = patch_sky.separation(object_coord)[flag]
    to_download.add_column(Column(data=distance.value, name='distance'))
    to_download.sort('distance')
    #to_download = to_download[:]
    print('# You have {} patches to be downloaded.'.format(len(to_download)))

    filenameset = []
    for obj in to_download:
        file = 'calexp_{0}_{1}_{2}.fits'.format(obj['tract'], obj['patch'].replace(',', '_'), band)
        filenameset.append(os.path.join(output_dir, file))
        if not os.path.isfile(file):
            pdr2.download_patch(obj['tract'], obj['patch'], filt='HSC-R', output_file=file);
            hdu = fits.open(os.path.join(output_dir, file))
            img = hdu[1].data
            hdr = hdu[1].header
            hdr['XTENSION'] = 'IMAGE'
            hdu.close()
            save_to_fits(img, os.path.join(output_dir, file), header=hdr);

    # Calculating image size in pixels
    imgsize = int(size.to(u.arcsec).value / DECaLS_pixel_scale)
    # Configure ``swarp``
    with open("config_swarp.sh","w+") as f:
        # check if swarp is installed
        f.write('for cmd in swarp; do\n')
        f.write('\t hasCmd=$(which ${cmd} 2>/dev/null)\n')
        f.write('\t if [[ -z "${hasCmd}" ]]; then\n')
        f.write('\t\t echo "This script requires ${cmd}, which is not in your \$PATH." \n')
        f.write('\t\t exit 1 \n')
        f.write('\t fi \n done \n\n')
        
        # Write ``default.swarp``.
        f.write('/bin/rm -f default.swarp \n')
        f.write('cat > default.swarp <<EOT \n')
        f.write('IMAGEOUT_NAME \t\t {}.fits      # Output filename\n'.format(os.path.join(output_dir, '_'.join([output_name, band]))))
        f.write('WEIGHTOUT_NAME \t\t {}_weights.fits     # Output weight-map filename\n\n'.format(os.path.join(output_dir, '_'.join([output_name, band]))))
        f.write('HEADER_ONLY            N               # Only a header as an output file (Y/N)?\nHEADER_SUFFIX          .head           # Filename extension for additional headers\n\n')
        f.write('#------------------------------- Input Weights --------------------------------\n\nWEIGHT_TYPE            NONE            # BACKGROUND,MAP_RMS,MAP_VARIANCE\n                                       # or MAP_WEIGHT\nWEIGHT_SUFFIX          weight.fits     # Suffix to use for weight-maps\nWEIGHT_IMAGE                           # Weightmap filename if suffix not used\n                                       # (all or for each weight-map)\n\n')
        f.write('#------------------------------- Co-addition ----------------------------------\n\nCOMBINE                Y               # Combine resampled images (Y/N)?\nCOMBINE_TYPE           MEDIAN          # MEDIAN,AVERAGE,MIN,MAX,WEIGHTED,CHI2\n                                       # or SUM\n\n')
        f.write('#-------------------------------- Astrometry ----------------------------------\n\nCELESTIAL_TYPE         NATIVE          # NATIVE, PIXEL, EQUATORIAL,\n                                       # GALACTIC,ECLIPTIC, or SUPERGALACTIC\nPROJECTION_TYPE        TAN             # Any WCS projection code or NONE\nPROJECTION_ERR         0.001           # Maximum projection error (in output\n                                       # pixels), or 0 for no approximation\nCENTER_TYPE            MANUAL          # MANUAL, ALL or MOST\n')
        f.write('CENTER   {0}, {1} # Image Center\n'.format(ra, dec))
        f.write('PIXELSCALE_TYPE        MANUAL          # MANUAL,FIT,MIN,MAX or MEDIAN\n')
        f.write('PIXEL_SCALE            {}  # Pixel scale\n'.format(DECaLS_pixel_scale))
        f.write('IMAGE_SIZE             {0},{1} # scale = 0.262 arcsec/pixel\n\n'.format(imgsize, imgsize))
        f.write('#-------------------------------- Resampling ----------------------------------\n\nRESAMPLE               Y               # Resample input images (Y/N)?\nRESAMPLE_DIR           .               # Directory path for resampled images\nRESAMPLE_SUFFIX        .resamp.fits    # filename extension for resampled images\n\nRESAMPLING_TYPE        LANCZOS3        # NEAREST,BILINEAR,LANCZOS2,LANCZOS3\n                                       # or LANCZOS4 (1 per axis)\nOVERSAMPLING           0               # Oversampling in each dimension\n                                       # (0 = automatic)\nINTERPOLATE            N               # Interpolate bad input pixels (Y/N)?\n                                       # (all or for each image)\n\nFSCALASTRO_TYPE        FIXED           # NONE,FIXED, or VARIABLE\nFSCALE_KEYWORD         FLXSCALE        # FITS keyword for the multiplicative\n                                       # factor applied to each input image\nFSCALE_DEFAULT         1.0             # Default FSCALE value if not in header\n\nGAIN_KEYWORD           GAIN            # FITS keyword for effect. gain (e-/ADU)\nGAIN_DEFAULT           0.0             # Default gain if no FITS keyword found\n\n')
        f.write('#--------------------------- Background subtraction ---------------------------\n\nSUBTRACT_BACK          N               # Subtraction sky background (Y/N)?\n                                       # (all or for each image)\n\nBACK_TYPE              AUTO            # AUTO or MANUAL\n                                       # (all or for each image)\nBACK_DEFAULT           0.0             # Default background value in MANUAL\n                                       # (all or for each image)\nBACK_SIZE              128             # Background mesh size (pixels)\n                                       # (all or for each image)\nBACK_FILTERSIZE        3               # Background map filter range (meshes)\n                                       # (all or for each image)\n\n')
        f.write('#------------------------------ Memory management -----------------------------\n\nVMEM_DIR               .               # Directory path for swap files\nVMEM_MAX               2047            # Maximum amount of virtual memory (MB)\nMEM_MAX                2048            # Maximum amount of usable RAM (MB)\nCOMBINE_BUFSIZE        1024            # Buffer size for combine (MB)\n\n')
        f.write('#------------------------------ Miscellaneous ---------------------------------\n\nDELETE_TMPFILES        Y               # Delete temporary resampled FITS files\n                                       # (Y/N)?\nCOPY_KEYWORDS          OBJECT          # List of FITS keywords to propagate\n                                       # from the input to the output headers\nWRITE_FILEINFO         Y               # Write information about each input\n                                       # file in the output image header?\nWRITE_XML              N               # Write XML file (Y/N)?\nXML_NAME               swarp.xml       # Filename for XML output\nVERBOSE_TYPE           QUIET           # QUIET,NORMAL or FULL\n\nNTHREADS               0               # Number of simultaneous threads for\n                                       # the SMP version of SWarp\n                                       # 0 = automatic \n')
        f.write('EOT\n')
        f.write('swarp ' + ' '.join(filenameset) + '\n\n')
        f.write('rm ' + os.path.join(output_dir, '_*'))
        f.close()
    
    os.system('/bin/bash config_swarp.sh')
    print('# The image is save as {}'.format(os.path.join(output_dir, '_'.join([output_name, band]))))

def download_sdss_large(ra, dec, band, size=0.7*u.deg, radius=0.5*u.deg, verbose=True,
                        output_dir='./', output_name='SDSS_large', overwrite=True):
    '''Download SDSS frames and stitch them together using ``swarp``. Hence ``swarp`` must be installed!
    ``swarp`` resamples the image, but doesn't correct background.

    Parameters:
        ra (float): RA of the object.
        dec (float): DEC of the object.
        band: string, such as 'r' or 'g'.
        size (``astropy.units`` object): size of cutout, it should be comparable to ``radius``.
        radius (``astropy.units`` object): bricks whose distances to the object are 
            nearer than this radius will be download.  
        output_dir (str): directory of output files.
        output_name (str): prefix of output images.
        overwrite (bool): overwrite files or not.
    
    Return:
        None
    '''
    import re
    import urllib
    from astropy.coordinates import SkyCoord
    from .utils import save_to_fits
    import subprocess

    URL = 'https://dr12.sdss.org/mosaics/script?onlyprimary=True&pixelscale=0.396&ra={0}&filters={2}&dec={1}&size={3}'.format(ra, dec, band, size.to(u.deg).value)
    urllib.request.urlretrieve(URL, filename='sdss_task.sh')
    with open('sdss_task.sh', 'r') as f:
        text = f.read()
        f.close()

    os.remove('sdss_task.sh')
    imgname = re.compile('IMAGEOUT_NAME\s*(\S*)').search(text).groups()[0]
    weightname = re.compile('WEIGHTOUT_NAME\s*(\S*)').search(text).groups()[0]
    text = text.replace(imgname, '_'.join([output_name, band]) + '.fits')
    text = text.replace(weightname, '_'.join([output_name, band, 'weight']) + '.fits')

    with open('sdss_task.sh', 'w+') as f:
        f.write(text)
        f.write('\n\n')
        f.write('rm frame*')
        f.close()

    a = os.system('/bin/bash sdss_task.sh')
    if a == 0:
        print('# The image is saved as {}.fits!'.format('_'.join([output_name, band])))

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
    
    return

    if high_res.lower() == 'hsc': 
        print(radius)
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


import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
from mrf.display import display_single
import copy
from astropy.stats import biweight_midvariance
from astropy.stats import biweight_location

def view_as_blocks(arr_in, block_shape):
    """
    Borrow from ``sklearn.utils``.
    Block view of the input n-dimensional array (using re-striding).

    Blocks are non-overlapping views of the input array.

    Parameters
    ----------
    arr_in : ndarray
        N-d input array.
    block_shape : tuple
        The shape of the block. Each dimension must divide evenly into the
        corresponding dimensions of `arr_in`.

    Returns
    -------
    arr_out : ndarray
        Block view of the input array.  If `arr_in` is non-contiguous, a copy
        is made.

    Examples
    --------
    >>> import numpy as np
    >>> from skimage.util.shape import view_as_blocks
    >>> A = np.arange(4*4).reshape(4,4)
    >>> A
    array([[ 0,  1,  2,  3],
           [ 4,  5,  6,  7],
           [ 8,  9, 10, 11],
           [12, 13, 14, 15]])
    >>> B = view_as_blocks(A, block_shape=(2, 2))
    >>> B[0, 0]
    array([[0, 1],
           [4, 5]])
    >>> B[0, 1]
    array([[2, 3],
           [6, 7]])
    >>> B[1, 0, 1, 1]
    13

    >>> A = np.arange(4*4*6).reshape(4,4,6)
    >>> A  # doctest: +NORMALIZE_WHITESPACE
    array([[[ 0,  1,  2,  3,  4,  5],
            [ 6,  7,  8,  9, 10, 11],
            [12, 13, 14, 15, 16, 17],
            [18, 19, 20, 21, 22, 23]],
           [[24, 25, 26, 27, 28, 29],
            [30, 31, 32, 33, 34, 35],
            [36, 37, 38, 39, 40, 41],
            [42, 43, 44, 45, 46, 47]],
           [[48, 49, 50, 51, 52, 53],
            [54, 55, 56, 57, 58, 59],
            [60, 61, 62, 63, 64, 65],
            [66, 67, 68, 69, 70, 71]],
           [[72, 73, 74, 75, 76, 77],
            [78, 79, 80, 81, 82, 83],
            [84, 85, 86, 87, 88, 89],
            [90, 91, 92, 93, 94, 95]]])
    >>> B = view_as_blocks(A, block_shape=(1, 2, 2))
    >>> B.shape
    (4, 2, 3, 1, 2, 2)
    >>> B[2:, 0, 2]  # doctest: +NORMALIZE_WHITESPACE
    array([[[[52, 53],
             [58, 59]]],
           [[[76, 77],
             [82, 83]]]])
    """
    from numpy.lib.stride_tricks import as_strided
    if not isinstance(block_shape, tuple):
        raise TypeError('block needs to be a tuple')

    block_shape = np.array(block_shape)
    if (block_shape <= 0).any():
        raise ValueError("'block_shape' elements must be strictly positive")

    if block_shape.size != arr_in.ndim:
        raise ValueError("'block_shape' must have the same length "
                         "as 'arr_in.shape'")

    arr_shape = np.array(arr_in.shape)
    if (arr_shape % block_shape).sum() != 0:
        raise ValueError("'block_shape' is not compatible with 'arr_in'")

    # -- restride the array to build the block view

    if not arr_in.flags.contiguous:
        warn(RuntimeWarning("Cannot provide views on a non-contiguous input "
                            "array without copying."))

    arr_in = np.ascontiguousarray(arr_in)

    new_shape = tuple(arr_shape // block_shape) + tuple(block_shape)
    new_strides = tuple(arr_in.strides * block_shape) + arr_in.strides

    arr_out = as_strided(arr_in, shape=new_shape, strides=new_strides)

    return arr_out

def block_reduce(image, block_size, func=np.sum, cval=0):
    """
    Borrow from ``sklearn.measure``.
    Down-sample image by applying function to local blocks.

    Parameters
    ----------
    image : ndarray
        N-dimensional input image.
    block_size : array_like
        Array containing down-sampling integer factor along each axis.
    func : callable
        Function object which is used to calculate the return value for each
        local block. This function must implement an ``axis`` parameter such
        as ``numpy.sum`` or ``numpy.min``.
    cval : float
        Constant padding value if image is not perfectly divisible by the
        block size.

    Returns
    -------
    image : ndarray
        Down-sampled image with same number of dimensions as input image.

    Examples
    --------
    >>> from skimage.measure import block_reduce
    >>> image = np.arange(3*3*4).reshape(3, 3, 4)
    >>> image # doctest: +NORMALIZE_WHITESPACE
    array([[[ 0,  1,  2,  3],
            [ 4,  5,  6,  7],
            [ 8,  9, 10, 11]],
           [[12, 13, 14, 15],
            [16, 17, 18, 19],
            [20, 21, 22, 23]],
           [[24, 25, 26, 27],
            [28, 29, 30, 31],
            [32, 33, 34, 35]]])
    >>> block_reduce(image, block_size=(3, 3, 1), func=np.mean)
    array([[[ 16.,  17.,  18.,  19.]]])
    >>> image_max1 = block_reduce(image, block_size=(1, 3, 4), func=np.max)
    >>> image_max1 # doctest: +NORMALIZE_WHITESPACE
    array([[[11]],
           [[23]],
           [[35]]])
    >>> image_max2 = block_reduce(image, block_size=(3, 1, 4), func=np.max)
    >>> image_max2 # doctest: +NORMALIZE_WHITESPACE
    array([[[27],
            [31],
            [35]]])
    """

    if len(block_size) != image.ndim:
        raise ValueError("`block_size` must have the same length "
                         "as `image.shape`.")

    pad_width = []
    for i in range(len(block_size)):
        if block_size[i] < 1:
            raise ValueError("Down-sampling factors must be >= 1. Use "
                             "`skimage.transform.resize` to up-sample an "
                             "image.")
        if image.shape[i] % block_size[i] != 0:
            after_width = block_size[i] - (image.shape[i] % block_size[i])
        else:
            after_width = 0
        pad_width.append((0, after_width))
    
    image = np.pad(image, pad_width=pad_width, mode='constant',
                   constant_values=cval)
    blocked = view_as_blocks(image, block_size)
    flatten = blocked.reshape((blocked.shape[0], blocked.shape[1], -1))

    return func(flatten, axis=(flatten.ndim - 1))

def cal_sblimit(image, mask, pixel_scale, zeropoint, scale_arcsec=60, minfrac=0.8, minback=6, verbose=True):
    import copy
    from astropy.stats import biweight_midvariance
    from astropy.stats import biweight_location
    
    print('### Determine surface brightness detection limit ###')
    ## read image
    ny, nx = image.shape
    if verbose:
        print('Dimensions of the input image: nx = {0}, ny = {1}'.format(nx, ny))
        print('Pixel scale of the input image: {0:.2f} arcsec/pix'.format(pixel_scale))
    
    scale_pix = scale_arcsec / pixel_scale # scale in pixel
    scale_x = np.array([scale_pix, int(scale_pix), int(scale_pix), int(scale_pix) + 1])
    scale_y = np.array([scale_pix, int(scale_pix), int(scale_pix) + 1, int(scale_pix) + 1])
    
    area = scale_x * scale_y
    area -= area[0]
    area = abs(area)[1:] # d1, d2, d3
    
    bin_x = int(scale_x[np.argmin(area) + 1])
    bin_y = int(scale_y[np.argmin(area) + 1])
    area_ratio = bin_x * bin_y / scale_pix**2 
    if verbose:
        print('Binning factors: dx = {0}, dy = {1}'.format(bin_x, bin_y))
        print('Used bin area / True bin area = {:.5f}'.format(area_ratio))
    
    nbins_x = np.int(nx / bin_x)
    nbins_y = np.int(ny / bin_y)

    im_var = np.zeros((nbins_y, nbins_x))
    im_loc = np.zeros((nbins_y, nbins_x))
    im_frac = np.zeros((nbins_y, nbins_x))
    im_fluct = np.zeros((nbins_y, nbins_x))

    
    for i in range(nbins_x - 1):
        for j in range(nbins_y - 1):
            x1, x2, y1, y2 = i * bin_x, (i + 1) * bin_x, j * bin_y, (j + 1) * bin_y
            im_sec = image[y1:y2, x1:x2]
            im_mask_sec = mask[y1:y2, x1:x2]
            im_sec_in = im_sec[(im_mask_sec == 0)]
            if im_sec_in.size > 0:
                im_var[j, i] = biweight_midvariance(im_sec_in)
                im_loc[j, i] = biweight_location(im_sec_in)
            im_frac[j, i] = 1 - np.float(im_sec_in.size) / np.float(im_sec.size)
            
    '''
    temp = copy.deepcopy(image)
    temp[mask==1] = np.nan
    # var
    im_var = block_reduce(temp, (bin_y, bin_x), func=np.nanvar, cval=np.nan)
    # loc
    im_loc = block_reduce(temp, (bin_y, bin_x), func=np.nanmedian, cval=np.nan)
    # frac
    im_frac = block_reduce(mask, (bin_y, bin_x), func=np.sum, cval=np.nan) / (bin_x * bin_y)
    '''

    # calculate fluctuation
    for i in range(1, nbins_x - 1):
        for j in range(1, nbins_y - 1):
            backvec = im_loc[j-1:j+2, i-1:i+2]
            backvec = np.delete(backvec.flatten(), 4)
            maskvec = im_frac[j-1:j+2, i-1:i+2]
            maskvec = np.delete(maskvec.flatten(), 4) # fraction of being masked out
            backvec_in = backvec[(maskvec < 1 - minfrac)]
            if len(backvec_in) > minback:
                im_fluct[j,i] = im_loc[j,i] - biweight_location(backvec_in)
            
    im_fluct_in = im_fluct[im_fluct != 0]
    sig_adu = np.sqrt(biweight_midvariance(im_fluct_in)) * 0.80  # 8/9 is area correction
    dsig_adu = sig_adu / np.sqrt(2 * (im_fluct_in.size - 1)) 
    # For the standard deviation of standard deviation, see this: https://stats.stackexchange.com/questions/631/standard-deviation-of-standard-deviation
    
    # convert to magnitudes
    sb_lim = zeropoint - 2.5 * np.log10(sig_adu / pixel_scale**2)
    dsb_lim = 2.5 / np.log(10) / sig_adu * dsig_adu
    if verbose:
        print('1-sigma variation in counts = {0:.4f} +- {1:.04f}'.format(sig_adu, dsig_adu))
        print('Surface brightness limit = {0:.4f} +- {1:.04f}'.format(sb_lim, dsb_lim))

    return sb_lim, sig_adu, [im_fluct, im_loc, im_var, im_frac]


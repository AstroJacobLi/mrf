import os
import copy
import numpy as np

from astropy import wcs
from astropy.io import fits
from astropy.table import Table, Column

import mrf

__all__ = ["imdelete", "imcopy", "imshift", "magnify", "blkavg"]


def imdelete(image, iraf_path=None):
    """
    Delete ``fits`` image.

    Parameters:
        image: str, name of the image to be deleted. Can also use regex (REGular EXpression).
        iraf_path: directory of "iraf/macosx/".

    Returns:
        None
    """
    if iraf_path is None:
        iraf_path = mrf.__file__.rstrip('__init__.py') + 'iraf/macosx/'
    if '*' not in image:
        if not os.path.isfile(image):
            raise ValueError("{} does not exist!".format(image))
    
    iraf_image = os.path.abspath(os.path.join(iraf_path, 'x_images.e'))
    Command = iraf_image + " imdelete "
    Command += 'images={} '.format(image)
    Command += 'verify={} '.format('no')
    os.system(Command)
    return 

def imcopy(image, output, iraf_path=None):
    """
    Copy ``fits`` image.

    Parameters:
        image: str, name of the image to be deleted. Can also use regex.
        output: str, name of output image.
        iraf_path: directory of "iraf/macosx/".
    
    Returns:
        None
    """
    if iraf_path is None:
        iraf_path = mrf.__file__.rstrip('__init__.py') + 'iraf/macosx/'
    if ':' not in image:
        if not os.path.isfile(image):
            raise ValueError("{} does not exist!".format(image))

    if os.path.isfile(output):
        imdelete(output, iraf_path=iraf_path)
    
    iraf_image = os.path.abspath(os.path.join(iraf_path, 'x_images.e'))
    Command = iraf_image + " imcopy "
    Command += 'input={} '.format(image)
    Command += 'output={} '.format(output)
    Command += 'verbose={} '.format('yes')
    os.system(Command)
    return 

def imshift(image, output, xshift, yshift, interp_type='poly3', boundary_type='constant', iraf_path=None):
    """
    Shift ``fits`` image, see https://iraf.net/irafhelp.php?val=imshift&help=Help+Page. 

    Parameters:
        image: str, name of the image to be deleted. Can also use regex.
        output: str, name of output image.
        xshift and yshift: float, shift pixels in x and y.
        interp_type: str, interpolant type. Available methods: 
            "nearest", "linear", "poly3", "poly5", "spline3", "sinc", "drizzle".
        boundary_type: str, available methods: "nearest", "constant", "reflect" and "wrap". 
            If use constant, fill the value with zero.
        iraf_path: directory of "iraf/macosx/".
    
    Returns:
        None
    """
    if iraf_path is None:
        iraf_path = mrf.__file__.rstrip('__init__.py') + 'iraf/macosx/'
    if not os.path.isfile(image):
        raise ValueError("{} does not exist!".format(image))
    if os.path.isfile(output):
        imdelete(output, iraf_path=iraf_path)

    iraf_image = os.path.abspath(os.path.join(iraf_path, 'x_images.e'))
    Command = iraf_image + " imshift "
    Command += 'input={} '.format(image)
    Command += 'output={} '.format(output)
    Command += 'xshift={} '.format(xshift)
    Command += 'yshift={} '.format(yshift)
    Command += 'shifts_file="" '
    Command += 'interp_type={} '.format(interp_type)
    Command += 'boundary_type={} '.format(boundary_type)
    if boundary_type == 'constant':
        Command += 'constant={} '.format(0.0)
    
    os.system(Command)
    return

def magnify(image, output, xmag, ymag, x1="INDEF", x2="INDEF", y1="INDEF", y2="INDEF", 
            interpolant='poly3', boundary='constant', fluxconserve='yes',
            logfile="STDOUT", iraf_path=None):
    """
    Magnify ``fits`` image, see https://iraf.net/irafhelp.php?val=imgeom.magnify&help=Help+Page. 

    Parameters:
        image: str, name of the image to be deleted. Can also use regex.
        output: str, name of output image.
        xmag(ymag): float, shift pixels in x and y.
        x1(x2): The starting and ending coordinates in x in the input image 
            which become the first and last pixel in x in the magnified image. 
            The values need not be integers. If indefinite the values default to 
            the first and last pixel in x of the input image; i.e. a value of 1 and nx.
        y1(y2): The starting and ending coordinates in y in the input image 
            which become the first and last pixel in y in the magnified image. 
            The values need not be integers. If indefinite the values default to 
            the first and last pixel in y of the input image; i.e. a value of 1 and ny.
        interp_type: str, interpolant type. Available methods: 
            "nearest", "linear", "poly3", "poly5", "spline3", "sinc", "lsinc", "drizzle".
        boundary_type: str, available methods: "nearest", "constant", "reflect" and "wrap". 
            If use constant, fill the value with zero.
        fluxconserve: str, whether conserve flux.
        iraf_path: directory of "iraf/macosx/".
    
    Returns:
        None
    """
    if iraf_path is None:
        iraf_path = mrf.__file__.rstrip('__init__.py') + 'iraf/macosx/'
    if not os.path.isfile(image):
        raise ValueError("{} does not exist!".format(image))
    if os.path.isfile(output):
        imdelete(output, iraf_path=iraf_path)

    iraf_image = os.path.abspath(os.path.join(iraf_path, 'x_images.e'))
    Command = iraf_image + " magnify "
    Command += 'input={} '.format(image)
    Command += 'output={} '.format(output)
    Command += 'xmag={} '.format(xmag)
    Command += 'ymag={} '.format(ymag)
    Command += 'x1={} '.format(x1)
    Command += 'x2={} '.format(x2)
    Command += 'y1={} '.format(y1)
    Command += 'y2={} '.format(y2)
    Command += 'dx=INDEF '
    Command += 'dy=INDEF '
    Command += 'interpolation={} '.format(interpolant)
    Command += 'boundary={} '.format(boundary)
    if boundary == 'constant':
        Command += 'constant={} '.format(0.0)
    Command += 'fluxconserve={} '.format(fluxconserve)
    Command += 'logfile={} '.format(logfile)
    os.system(Command)
    return

def blkavg(image, output, b1, b2, option="sum", iraf_path=None):
    """
    Block average ``fits`` image, see https://iraf.net/irafhelp.php?val=imgeom.magnify&help=Help+Page.

    Parameters:
        image: str, name of the image to be deleted. Can also use regex.
        output: str, name of output image.
        b1: block size (column).
        b2: block size (row).
        option: str, choices are "sum" and "average".
    
    Returns:
        None
    """
    if iraf_path is None:
        iraf_path = mrf.__file__.rstrip('__init__.py') + 'iraf/macosx/'
    if not os.path.isfile(image):
        raise ValueError("{} does not exist!".format(image))
    if os.path.isfile(output):
        imdelete(output, iraf_path=iraf_path)

    iraf_image = os.path.abspath(os.path.join(iraf_path, 'x_images.e'))
    Command = iraf_image + " blkavg "
    Command += 'input={} '.format(image)
    Command += 'output={} '.format(output)
    Command += 'b1={} '.format(b1)
    Command += 'b2={} '.format(b2)
    Command += 'option={} '.format(option)
    os.system(Command)
    return
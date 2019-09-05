# -*- coding: utf-8 -*-
import warnings
warnings.simplefilter('ignore')

__all__ = ["utils", "display", "celestial", "imtools", "task", "download"]

# Version
__version__ = "1.0.1"
__name__ = 'mrf'

# Define pixel scale of different surveys, unit = arcsec / pixel
HSC_pixel_scale = 0.168
DECaLS_pixel_scale = 0.262
Dragonfly_pixel_scale = 2.5
SDSS_pixel_scale = 0.395
CFHT_pixel_scale = 0.186

# Define zeropoint of different surveys
HSC_zeropoint = 27.0
DECaLS_zeropoint = 22.5
SDSS_zeropoint = 22.5
CFHT_zeropoint = 30.0

# Define FWHM seeing of different surveys
HSC_seeing = 0.7
Dragonfly_seeing = 6.0

# Star catalogs in VizieR
USNO_vizier = 'I/252/out'
APASS_vizier = 'II/336'

HSC_binray_mask_dict = {0: 'BAD',
                        1:  'SAT (saturated)',
                        2:  'INTRP (interpolated)',
                        3:  'CR (cosmic ray)',
                        4:  'EDGE (edge of the CCD)',
                        5:  'DETECTED',
                        6:  'DETECTED_NEGATIVE',
                        7:  'SUSPECT (suspicious pixel)',
                        8:  'NO_DATA',
                        9:  'BRIGHT_OBJECT (bright star mask, not available in S18A yet)',
                        10: 'CROSSTALK', 
                        11: 'NOT_DEBLENDED (For objects that are too big to run deblender)',
                        12: 'UNMASKEDNAN',
                        13: 'REJECTED',
                        14: 'CLIPPED',
                        15: 'SENSOR_EDGE',
                        16: 'INEXACT_PSF'}

filter_corr_dict = {'df-des': {'r': 0.13, 'g': 0.05},
                    'df-cfht': {'r': 0.06, 'g': 0.10},
                    'df-kids': {'r': 0.06, 'g': -0.01}
                   }


filter_corr_synthetic_dict = {'df-des': {'r': 0.07, 'g': 0.03},
                              'df-cfht': {'r': 0.00, 'g': 0.05},
                             }
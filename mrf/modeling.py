#########################################################################
################### From Qing Liu (UToronto) ############################
#########################################################################
import math
import numpy as np
import matplotlib.pyplot as plt

import galsim
from galsim import GalSimBoundsError

from scipy.integrate import quad
from scipy.spatial import distance
from scipy.special import gamma as Gamma

from astropy.io import fits
from astropy.utils import lazyproperty

from itertools import combinations
from functools import partial, lru_cache

from copy import deepcopy

############################################
# Functions for making PSF models
############################################

class PSF_Model:
    """ A PSF Model object """
    
    def __init__(self, params=None,
                 core_model='moffat',
                 aureole_model='power'):
        """
        Parameters
        ----------
        params : a dictionary containing keywords of PSF parameter
        core_model : model of PSF core (moffat)
        aureole_model : model of aureole ("moffat, "power" or "multi-power")    
        
        """
        self.core_model = core_model
        self.aureole_model = aureole_model
        
        self.params = params
        
        # Build attribute for parameters from dictionary keys 
        for key, val in params.items():
            exec('self.' + key + ' = val')
            
        if hasattr(self, 'fwhm'):
            self.gamma = fwhm_to_gamma(self.fwhm, self.beta)
            self.params['gamma'] = self.gamma
            
        if hasattr(self, 'gamma'):
            self.fwhm  = gamma_to_fwhm(self.gamma, self.beta)
            self.params['fwhm'] = self.fwhm
            
        self.gsparams = galsim.GSParams(folding_threshold=1e-10)
        
    def __str__(self):
        return "A PSF Model Class"

    def __repr__(self):
        return " ".join([f"{self.__class__.__name__}", f"<{self.aureole_model}>"])
                
    def pixelize(self, pixel_scale=2.5):
        """ Build grid for drawing """
        self.pixel_scale = pixel_scale
        
        for key, val in self.params.items():
            if ('gamma' in key) | ('theta' in key):
                val = val / pixel_scale
                exec('self.' + key + '_pix' + ' = val')
                
    def update(self, params):
        """ Update PSF parameters from dictionary keys """
        pixel_scale = self.pixel_scale
        for key, val in params.items():
            if np.ndim(val) > 0:
                val = np.array(val)
                
            exec('self.' + key + ' = val')
            self.params[key] = val
            
            if ('gamma' in key) | ('theta' in key):
                val = val / pixel_scale
                exec('self.' + key + '_pix' + ' = val')
                
    def copy(self):
        """ A deep copy of the object """
        return deepcopy(self)            

    @property
    def f_core1D(self):
        """ 1D Core function """
        gamma_pix, beta = self.gamma_pix, self.beta
        c_mof2Dto1D = C_mof2Dto1D(gamma_pix, beta)
        return lambda r: moffat1d_normed(r, gamma_pix, beta) / c_mof2Dto1D

    @property
    def f_aureole1D(self):
        """ 1D Aureole function """
        if self.aureole_model == "moffat":
            gamma1_pix, beta1 = self.gamma1_pix, self.beta1
            c_mof2Dto1D = C_mof2Dto1D(gamma1_pix, beta1)
            f_aureole = lambda r: moffat1d_normed(r, gamma1_pix, beta1) / c_mof2Dto1D
        
        elif self.aureole_model == "power":
            n0, theta_0_pix = self.n0, self.theta_0_pix
            c_aureole_2Dto1D = C_pow2Dto1D(n0, theta_0_pix)
            f_aureole = lambda r: trunc_power1d_normed(r, n0, theta_0_pix) / c_aureole_2Dto1D

        elif self.aureole_model == "multi-power":
            n_s, theta_s_pix = self.n_s, self.theta_s_pix
            c_aureole_2Dto1D = C_mpow2Dto1D(n_s, theta_s_pix)
            f_aureole = lambda r: multi_power1d_normed(r, n_s, theta_s_pix) / c_aureole_2Dto1D

        return f_aureole


    def plot1D(self, **kwargs):
        """ Plot 1D profile """
        from .display import plot_PSF_model_1D
        
        plot_PSF_model_1D(self.frac, self.f_core1D, self.f_aureole1D, **kwargs)
        
        if self.aureole_model == "multi-power":
            for t in self.theta_s_pix:
                plt.axvline(t, ls="--", color="k", alpha=0.3, zorder=1)
                
    def generate_core(self):
        """ Generate Galsim PSF of core. """
        gamma, beta = self.gamma, self.beta
        self.fwhm = fwhm = gamma * 2. * math.sqrt(2**(1./beta)-1)
        
        psf_core = galsim.Moffat(beta=beta, fwhm=fwhm,
                                 flux=1., gsparams=self.gsparams) # in arcsec
        self.psf_core = psf_core
        return psf_core
    
    def generate_aureole(self,
                         contrast=1e6,
                         psf_scale=None,
                         psf_range=None,
                         min_psf_range=60,
                         max_psf_range=720,
                         interpolant="cubic"):
        """
        Generate Galsim PSF of aureole.

        Parameters
        ----------
        contrast: Ratio of the intensity at max range and at center. Used to calculate the PSF size if not given.
        psf_scale: Pixel scale of the PSF, <= pixel scale of data. In arcsec/pix.
        psf_range: Range of PSF. In arcsec.
        min_psf_range : Minimum range of PSF. In arcsec.
        max_psf_range : Maximum range of PSF. In arcsec.
        interpolant: Interpolant method in Galsim.
        
        Returns
        ----------
        psf_aureole: power law Galsim PSF, flux normalized to be 1.
        psf_size: Full image size of PSF used. In pixel.
        
        """
        
        if psf_scale is None:
            psf_scale = self.pixel_scale
            
        if self.aureole_model == "moffat":
            gamma1, beta1 = self.gamma1, self.beta1
            
            if psf_range is None:
                psf_range = max_psf_range
            psf_size = round_good_fft(2 * psf_range // psf_scale)   
    
        else:
            if self.aureole_model == "power":
                n0 = self.n0
                theta_0 = self.theta_0

            elif self.aureole_model == "multi-power":
                n_s = self.n_s
                theta_s = self.theta_s
                self.n0 = n0 = n_s[0]
                self.theta_0 = theta_0 = theta_s[0]

            if psf_range is None:
                psf_size = calculate_psf_size(n0, theta_0, contrast,
                                              psf_scale, min_psf_range, max_psf_range)
            else:
                psf_size = round_good_fft(psf_range) 
        
            # Generate Grid of PSF and plot PSF model in real space onto it
            xx_psf, yy_psf, cen_psf = generate_psf_grid(psf_size)
        
        if self.aureole_model == "moffat":
            psf_aureole = galsim.Moffat(beta=beta1, scale_radius=gamma1,
                                        flux=1., gsparams=self.gsparams)
            
        else:
            if self.aureole_model == "power":
                theta_0_pix = theta_0 / psf_scale
                psf_model = trunc_power2d(xx_psf, yy_psf,
                                          n0, theta_0_pix, I_theta0=1, cen=cen_psf) 

            elif self.aureole_model == "multi-power":
                theta_s_pix = theta_s / psf_scale
                psf_model =  multi_power2d(xx_psf, yy_psf,
                                           n_s, theta_s_pix, 1, cen=cen_psf) 

            # Parse the image to Galsim PSF model by interpolation
            image_psf = galsim.ImageF(psf_model)
            psf_aureole = galsim.InterpolatedImage(image_psf, flux=1,
                                                   scale=psf_scale,
                                                   x_interpolant=interpolant,
                                                   k_interpolant=interpolant)
        self.psf_aureole = psf_aureole
        return psf_aureole, psf_size   

        
    def Flux2Amp(self, Flux):
        """ Convert Flux to Astropy Moffat Amplitude (pixel unit) """
        
        Amps = [moffat2d_Flux2Amp(self.gamma_pix, self.beta, Flux=(1-self.frac)*F)
                for F in Flux]
        return np.array(Amps)
    
    
    def I2I0(self, I, r=12):
        """ Convert aureole I(r) at r to I0. r in pixel """
        
        if self.aureole_model == "moffat":
            return I2I0_mof(self.gamma1_pix, self.beta1, r, I=I)
        
        elif self.aureole_model == "power":
            return I2I0_pow(self.n0, self.theta_0_pix, r, I=I)
        
        elif self.aureole_model == "multi-power":
            return I2I0_mpow(self.n_s, self.theta_s_pix, r, I=I)
        
    def I02I(self, I0, r=12):
        """ Convert aureole I(r) at r to I0. r in pixel """
        
        if self.aureole_model == "moffat":
            return I02I_mof(self.gamma1_pix, self.beta1, r, I0=I0)
        
        elif self.aureole_model == "power":
            return I02I_pow(self.n0, self.theta_0_pix, r, I0=I0)
        
        elif self.aureole_model == "multi-power":
            return I02I_mpow(self.n_s, self.theta_s_pix, r, I0=I0)
    
    def calculate_external_light(self, stars, n_iter=2):
        """ Calculate the integrated external scatter light that affects
        the flux scaling from very bright stars on the other stars.
        
        Parameters
        ----------
        stars : Star object
        n_iter : iteration time to do the calculation
        
        """
        
        I_ext = np.zeros(stars.n_bright)
        
        if self.aureole_model == "moffat":
            pass
        
        else:
            z_norm_verybright0 = stars.z_norm_verybright.copy()
            pos_source, pos_eval = stars.star_pos_verybright, stars.star_pos_bright

            if self.aureole_model == "power":
                cal_ext_light = partial(calculate_external_light_pow,
                                       n0=self.n0, theta0=self.theta_0_pix,
                                       pos_source=pos_source, pos_eval=pos_eval)
            elif self.aureole_model == "multi-power":
                cal_ext_light = partial(calculate_external_light_mpow,
                                        n_s=self.n_s, theta_s_pix=self.theta_s_pix,
                                        pos_source=pos_source, pos_eval=pos_eval)
            # Loop the subtraction    
            r_scale = stars.r_scale
            n_verybright = stars.n_verybright
            for i in range(n_iter):
                z_norm_verybright = z_norm_verybright0 - I_ext[:n_verybright]
                I0_verybright = self.I2I0(z_norm_verybright, r=r_scale)
                I_ext = cal_ext_light(I0_source=I0_verybright)
            
        return I_ext
    
    def I2Flux(self, I, r=12):
        """ Convert aureole I(r) at r to total flux. r in pixel """
        
        if self.aureole_model == "moffat":
            return I2Flux_mof(self.frac, self.gamma1_pix, self.beta1, r, I=I)
        
        elif self.aureole_model == "power":
            return I2Flux_pow(self.frac, self.n0, self.theta_0_pix, r, I=I)
        
        elif self.aureole_model == "multi-power":
            return I2Flux_mpow(self.frac, self.n_s, self.theta_s_pix, r, I=I)
        
    def Flux2I(self, Flux, r=12):
        """ Convert aureole I(r) at r to total flux. r in pixel """
        
        if self.aureole_model == "moffat":
            return Flux2I_mof(self.frac, self.gamma1_pix, self.beta1, r, Flux=Flux)
        
        elif self.aureole_model == "power":
            return Flux2I_pow(self.frac, self.n0, self.theta_0_pix, r, Flux=Flux)
        
        elif self.aureole_model == "multi-power":
            return Flux2I_mpow(self.frac, self.n_s, self.theta_s_pix, r,  Flux=Flux)
        
    def SB2Flux(self, SB, BKG, ZP, r=12):
        """ Convert suface brightness SB at r to total flux, given background value and ZP. """
        # Intensity = I + BKG
        I = SB2Intensity(SB, BKG, ZP, self.pixel_scale) - BKG
        return self.I2Flux(I, r=r)
    
    def Flux2SB(self, Flux, BKG, ZP, r=12):
        """ Convert total flux to suface brightness SB at r, given background value and ZP. """
        I = self.Flux2I(Flux, r=r)
        return Intensity2SB(I+ BKG, BKG, ZP, self.pixel_scale)
    
    @property
    def psf_star(self):
        """ Galsim object of star psf (core+aureole) """
        frac = self.frac
        psf_core, psf_aureole = self.psf_core, self.psf_aureole
        return (1-frac) * psf_core + frac * psf_aureole

    def plot_PSF_model_galsim(self, contrast=None, save=False, save_dir='.'):
        """ Build and plot Galsim 2D model averaged in 1D """
        from .display import plot_PSF_model_galsim
        image_psf = plot_PSF_model_galsim(self, contrast=contrast,
                                          save=save, save_dir=save_dir)
        self.image_psf = image_psf
    
    @staticmethod
    def write_psf_image(image_psf, filename='PSF_model.fits'):
        """ Write the 2D psf image to fits """
        hdu = fits.ImageHDU(image_psf)
        hdu.writeto(filename, overwrite=True)
    
    def draw_core2D_in_real(self, star_pos, Flux):
        """ 2D drawing function of the core in real space given positions and flux (of core) of target stars """
        
        gamma, alpha = self.gamma_pix, self.beta
        Amps = np.array([moffat2d_Flux2Amp(gamma, alpha, Flux=flux)
                       for flux in Flux])
        f_core_2d_s = np.array([models.Moffat2D(amplitude=amp, x_0=x0, y_0=y0,
                                                gamma=gamma, alpha=alpha)
                                for ((x0,y0), amp) in zip(star_pos, Amps)])
            
        return f_core_2d_s
        
    def draw_aureole2D_in_real(self, star_pos, Flux=None, I0=None):
        """ 2D drawing function of the aureole in real space given positions and flux / amplitude (of aureole) of target stars """
        
        if self.aureole_model == "moffat":
            gamma1_pix, alpha1 = self.gamma1_pix, self.beta1
            
            # In this case I_theta0 is defined as the amplitude at gamma
            if I0 is None:
                I_theta0 = moffat2d_Flux2I0(gamma1_pix, alpha1, Flux=Flux)
            elif Flux is None:
                I_theta0 = I0
            else:
                raise MyError("Both Flux and I0 are not given.")
                
            Amps = np.array([moffat2d_I02Amp(alpha1, I0=I0)
                             for I0 in I_theta0])
            
            f_aureole_2d_s = np.array([models.Moffat2D(amplitude=amp,
                                                       x_0=x0, y_0=y0,
                                                       gamma=gamma1_pix,
                                                       alpha=alpha1)
                                    for ((x0,y0), amp) in zip(star_pos, Amps)])
            
        elif self.aureole_model == "power":
            n0 = self.n0
            theta_0_pix = self.theta_0_pix
            
            if I0 is None:
                I_theta0 = power2d_Flux2Amp(n0, theta_0_pix, Flux=1) * Flux
            elif Flux is None:
                I_theta0 = I0
            else:
                raise MyError("Both Flux and I0 are not given.")
            
            f_aureole_2d_s = np.array([lambda xx, yy, cen=pos, I=I:\
                                      trunc_power2d(xx, yy, cen=cen,
                                                    n=n0, theta0=theta_0_pix,
                                                    I_theta0=I)
                                      for (I, pos) in zip(I_theta0, star_pos)])

        elif self.aureole_model == "multi-power":
            n_s = self.n_s
            theta_s_pix = self.theta_s_pix
            
            if I0 is None:
                I_theta0 = multi_power2d_Flux2Amp(n_s, theta_s_pix, Flux=1) * Flux
            elif Flux is None:
                I_theta0 = I0
            else:
                raise MyError("Both Flux and I0 are not given.")
            
            f_aureole_2d_s = np.array([lambda xx, yy, cen=pos, I=I:\
                                      multi_power2d(xx, yy, cen=cen,
                                                    n_s=n_s, theta_s=theta_s_pix,
                                                    I_theta0=I)
                                      for (I, pos) in zip(I_theta0, star_pos)])
            
        return f_aureole_2d_s

    
############################################
# Analytic Functions for models
############################################

### Baisc Funcs ###

def fwhm_to_gamma(fwhm, beta):
    """ in arcsec """
    return fwhm / 2. / np.sqrt(2**(1. / beta) - 1)

def gamma_to_fwhm(gamma, beta):
    """ in arcsec """
    return gamma / fwhm_to_gamma(1, beta)
    
def Intensity2SB(I, BKG, ZP, pixel_scale=2.5):
    """ Convert intensity to surface brightness (mag/arcsec^2) given the background value, zero point and pixel scale """
    I = np.atleast_1d(I)
    I[np.isnan(I)] = BKG
    if np.any(I <= BKG):
        I[I <= BKG] = np.nan
    I_SB = -2.5 * np.log10(I - BKG) + ZP + 2.5 * np.log10(pixel_scale**2)
    return I_SB

def SB2Intensity(SB, BKG, ZP, pixel_scale=2.5):
    """ Convert surface brightness (mag/arcsec^2)to intensity given the background value, zero point and pixel scale """ 
    SB = np.atleast_1d(SB)
    I = 10**((SB - ZP - 2.5 * np.log10(pixel_scale**2))/ (-2.5)) + BKG
    return I

def round_good_fft(x):
    # Rounded PSF size to 2^k or 3*2^k
    a = 1 << int(x-1).bit_length()
    b = 3 << int(x-1).bit_length()-2
    if x>b:
        return a
    else:
        return min(a,b)

@lru_cache(maxsize=16)
def generate_psf_grid(psf_size):
    # Generate Grid of PSF and plot PSF model in real space onto it
    cen_psf = ((psf_size-1)/2., (psf_size-1)/2.)
    yy_psf, xx_psf = np.mgrid[:psf_size, :psf_size]
    return xx_psf, yy_psf, cen_psf

### funcs on single element ###

def trunc_pow(x, n, theta0, I_theta0=1):
    """ Truncated power law for single element, I = I_theta0 at theta0 """
    a = I_theta0 / (theta0)**(-n)
    y = a * x**(-n) if x > theta0 else I_theta0
    return y

def multi_pow(x, n_s, theta_s, I_theta0, a_s=None):
    """ Continuous multi-power law for single element """
    
    if a_s is None:
        a_s = compute_multi_pow_norm(n_s, theta_s, I_theta0)
    n0, theta0, a0 = n_s[0], theta_s[0], a_s[0]
    
    if x <= theta0:
        return I_theta0
    elif x<= theta_s[1]:
        y = a0 * x**(-n0)
        return y
    else:
        for k in range(len(a_s)):
            try:
                if x <= theta_s[k+2]:
                    y = a_s[k+1] * x**(-n_s[k+1])
                    return y
            except IndexError:
                pass
        else:
            y = a_s[-1] * x**(-n_s[-1])
            return y


### 1D functions ###

def power1d(x, n, theta0, I_theta0):
    """ Power law for 1d array, I = I_theta0 at theta0, theta in pix """
    a = I_theta0 / (theta0)**(-n)
    y = a * np.power(x + 1e-6, -n)
    return y

def multi_power1d(x, n_s, theta_s, I_theta0):
    """ Multi-power law for 1d array, I = I_theta0 at theta0, theta in pix"""
    a_s = compute_multi_pow_norm(n_s, theta_s, I_theta0)
    theta0 = theta_s[0]
    
    y = np.zeros_like(x)
    y[x<=theta0] = I_theta0
    
    for k in range(len(a_s)):
        reg = (x>theta_s[k]) & (x<=theta_s[k+1]) if k<len(a_s)-1 else (x>theta_s[k])  
        y[reg] = a_s[k] * np.power(x[reg], -n_s[k])
    return y

def trunc_power1d(x, n, theta0, I_theta0=1): 
    """ Truncated power law for 1d array, I = I_theta0 at theta0, theta in pix """
    a = I_theta0 / (theta0)**(-n)
    y = a * np.power(x + 1e-6, -n) 
    y[x<=theta0] = I_theta0
    return y


def compute_multi_pow_norm(n_s, theta_s, I_theta0):
    """ Compute normalization factor A of each power law component A_i*(theta)^(n_i)"""
    n0, theta0 = n_s[0], theta_s[0]
    a0 = I_theta0 * theta0**(n0)
    a_s = np.zeros(len(n_s))   
    a_s[0] = a0
    
    I_theta_i = a0 * float(theta_s[1])**(-n0)
    for i, (n_i, theta_i) in enumerate(zip(n_s[1:], theta_s[1:])):
        a_i = I_theta_i/(theta_s[i+1])**(-n_i)
        try:
            a_s[i+1] = a_i
            I_theta_i = a_i * float(theta_s[i+2])**(-n_i)
        except IndexError:
            pass 
    return a_s


def moffat_power1d(x, gamma, alpha, n, theta0, A=1):
    """ Moffat + Power for 1d array, flux normalized = 1, theta in pix """
    Mof_mod_1d = models.Moffat1D(amplitude=A, x_0=0, gamma=gamma, alpha=alpha)
    y[x<=theta0] = Mof_mod_1d(x)
    y[x>theta0] = power1d(x[x>theta0], n, theta0, Mof_mod_1d(theta0))
    return y


def trunc_power1d_normed(x, n, theta0):
    """ Truncated power law for 1d array, flux normalized = 1, theta in pix """
    norm_pow = quad(trunc_pow, 0, np.inf, args=(n, theta0, 1))[0]
    y = trunc_power1d(x, n, theta0, 1) / norm_pow  
    return y


def moffat1d_normed(x, gamma, alpha):
    """ Moffat for 1d array, flux normalized = 1 """
    from astropy.modeling.models import Moffat1D
    Mof_mod_1d = Moffat1D(amplitude=1, x_0=0, gamma=gamma, alpha=alpha)
    norm_mof = quad(Mof_mod_1d, 0, np.inf)[0] 
    y = Mof_mod_1d(x) / norm_mof
    return y


def multi_power1d_normed(x, n_s, theta_s):
    """ Multi-power law for 1d array, flux normalized = 1, theta in pix """
    a_s = compute_multi_pow_norm(n_s, theta_s, 1)
    norm_mpow = quad(multi_pow, 0, np.inf,
                     args=(n_s, theta_s, 1, a_s), limit=100)[0]
    y = multi_power1d(x, n_s, theta_s, 1) / norm_mpow
    return y


### 2D functions ###

def power2d(xx, yy, n, theta0, I_theta0, cen): 
    """ Power law for 2d array, normalized = I_theta0 at theta0 """
    rr = np.sqrt((xx-cen[0])**2 + (yy-cen[1])**2) + 1e-6
    rr[rr<=1] = rr[rr>1].min()
    a = I_theta0 / (theta0)**(-n)
    z = a * np.power(rr, -n) 
    return z 


def trunc_power2d(xx, yy, n, theta0, I_theta0, cen): 
    """ Truncated power law for 2d array, normalized = I_theta0 at theta0 """
    rr = np.sqrt((xx-cen[0])**2 + (yy-cen[1])**2) + 1e-6
    a = I_theta0 / (theta0)**(-n)
    z = a * np.power(rr, -n) 
    z[rr<=theta0] = I_theta0
    return z


def multi_power2d(xx, yy, n_s, theta_s, I_theta0, cen):
    """ Multi-power law for 2d array, I = I_theta0 at theta0, theta in pix"""
    a_s = compute_multi_pow_norm(n_s, theta_s, I_theta0)
    
    rr = np.sqrt((xx-cen[0])**2 + (yy-cen[1])**2).ravel()
    z = np.zeros(xx.size) 
    theta0 = theta_s[0]
    z[rr<=theta0] = I_theta0
    
    for k in range(len(a_s)):
        reg = (rr>theta_s[k]) & (rr<=theta_s[k+1]) if k<len(a_s)-1 else (rr>theta_s[k])     
        z[reg] = a_s[k] * np.power(rr[reg], -n_s[k])
        
    return z.reshape(xx.shape)

### Flux/Amplitude Convertion ###

def moffat1d_Flux2Amp(r_core, beta, Flux=1):
    """ Calculate the (astropy) amplitude of 1d Moffat profile given the core width, power index, and total flux F.
    Note in astropy unit (x,y) the amplitude should be scaled with 1/sqrt(pi)."""
    Amp = Flux * Gamma(beta) / ( r_core * np.sqrt(np.pi) * Gamma(beta-1./2) ) # Derived scaling factor
    return  Amp

def moffat1d_Amp2Flux(r_core, beta, Amp=1):
    Flux = Amp / moffat1d_Flux2Amp(r_core, beta, Flux=1)
    return  Flux

def power1d_Flux2Amp(n, theta0, Flux=1, trunc=True):
    if trunc:
        I_theta0 = Flux * (n-1)/n / theta0
    else:
        I_theta0 = Flux * (n-1) / theta0
    return I_theta0

def power1d_Amp2Flux(n, theta0, Amp=1, trunc=True):
    if trunc:
        Flux = Amp * n/(n-1) * theta0
    else:
        Flux = Amp * 1./(n-1) * theta0
    return Flux

def moffat2d_Flux2Amp(r_core, beta, Flux=1):
    return Flux * (beta-1) / r_core**2 / np.pi

def moffat2d_Amp2Flux(r_core, beta, Amp=1):
    return Amp / moffat2d_Flux2Amp(r_core, beta, Flux=1)

def moffat2d_Flux2I0(r_core, beta, Flux=1):
    Amp = moffat2d_Flux2Amp(r_core, beta, Flux=Flux)
    return moffat2d_Amp2I0(beta, Amp=Amp)

def moffat2d_I02Amp(beta, I0=1):
    # Convert I0(r=r_core) to Amplitude
    return I0 * 2**(2*beta)

def moffat2d_Amp2I0(beta, Amp=1):
    # Convert I0(r=r_core) to Amplitude
    return Amp * 2**(-2*beta)


def power2d_Flux2Amp(n, theta0, Flux=1):
    if n>2:
        I_theta0 = (1./np.pi) * Flux * (n-2)/n / theta0**2
    else:
        raise InconvergenceError('PSF is not convergent in Infinity.')
        
    return I_theta0

def power2d_Amp2Flux(n, theta0, Amp=1):
    return Amp / power2d_Flux2Amp(n, theta0, Flux=1)
            
def multi_power2d_Amp2Flux(n_s, theta_s, Amp=1, theta_trunc=1e5):
    """ convert amplitude(s) to integral flux with 2D multi-power law """
    if np.ndim(Amp)>0:
        a_s = compute_multi_pow_norm(n_s, theta_s, 1)
        a_s = np.multiply(a_s[:,np.newaxis], Amp)
    else:
        a_s = compute_multi_pow_norm(n_s, theta_s, Amp)

    I_2D = sum_I2D_multi_power2d(Amp, a_s, n_s, theta_s, theta_trunc)
        
    return I_2D

def sum_I2D_multi_power2d(Amp, a_s, n_s, theta_s, theta_trunc=1e5):
    """ Supplementary function for multi_power2d_Amp2Flux tp speed up """
    
    theta0 = theta_s[0]
    I_2D = Amp * np.pi * theta0**2

    for k in range(len(n_s)-1):

        if n_s[k] == 2:
            I_2D += 2*np.pi * a_s[k] * math.log(theta_s[k+1]/theta_s[k])
        else:
            I_2D += 2*np.pi * a_s[k] * (theta_s[k]**(2-n_s[k]) - theta_s[k+1]**(2-n_s[k])) / (n_s[k]-2)

    if n_s[-1] > 2:
        I_2D += 2*np.pi * a_s[-1] * theta_s[-1]**(2-n_s[-1]) / (n_s[-1]-2) 
    elif n_s[-1] == 2:
        I_2D += 2*np.pi * a_s[-1] * math.log(theta_trunc/theta_s[-1])
    else:
        I_2D += 2*np.pi * a_s[-1] * (theta_trunc**(2-n_s[-1]) - theta_s[-1]**(2-n_s[-1])) / (2-n_s[-1])
        
    return I_2D

def multi_power2d_Flux2Amp(n_s, theta_s, Flux=1):
    return Flux / multi_power2d_Amp2Flux(n_s, theta_s, Amp=1)


def I2I0_mof(r_core, beta, r, I=1):
    """ Convert Intensity I(r) at r to I at r_core with moffat.
        r_core and r in pixel """
    Amp = I * (1+(r/r_core)**2)**beta
    I0 = moffat2d_Amp2I0(beta, Amp)
    return I0

def I02I_mof(r_core, beta, r, I0=1):
    """ Convert I at r_core to Intensity I(r) at r with moffat.
        r_core and r in pixel """
    Amp = moffat2d_I02Amp(beta, I0)
    I = Amp * (1+(r/r_core)**2)**(-beta)
    return I

def I2Flux_mof(frac, r_core, beta, r, I=1):
    """ Convert Intensity I(r) at r to total flux with fraction of moffat.
        r_core and r in pixel """
    Amp = I * (1+(r/r_core)**2)**beta
    Flux_mof = moffat2d_Amp2Flux(r_core, beta, Amp=Amp)
    Flux_tot = Flux_mof / frac
    return Flux_tot

def Flux2I_mof(frac, r_core, beta, r, Flux=1):
    """ Convert total flux  at r to Intensity I(r) with fraction of moffat.
        r_core and r in pixel """
    Flux_mof = Flux * frac
    Amp = moffat2d_Flux2Amp(r_core, beta, Flux=Flux_mof)
    I = Amp * (1+(r/r_core)**2)**(-beta)
    return I


def I2I0_pow(n0, theta0, r, I=1):
    """ Convert Intensity I(r) at r to I at theta_0 with power law.
        theata_s and r in pixel """
    I0 = I * (r/theta0)**n0
    return I0

def I02I_pow(n0, theta0, r, I0=1):
    """ Convert Intensity I(r) at r to I at theta_0 with power law.
        theata_s and r in pixel """
    I = I0 / (r/theta0)**n0
    return I

def I2Flux_pow(frac, n0, theta0, r, I=1):
    """ Convert Intensity I(r) at r to total flux with fraction of power law.
        theata0 and r in pixel """
    I0 = I2I0_pow(n0, theta0, r, I=I)
    Flux_pow = power2d_Amp2Flux(n0, theta0, Amp=I0)
    Flux_tot = Flux_pow / frac
    return Flux_tot

def Flux2I_pow(frac, n0, theta0, r, Flux=1):
    """ Convert total flux to Intensity I(r) at r.
        theata0 and r in pixel """
    Flux_pow = Flux * frac
    I0 = power2d_Flux2Amp(n0, theta0, Flux=Flux_pow)
    I = I0 / (r/theta0)**n0
    return I

def I2I0_mpow(n_s, theta_s_pix, r, I=1):
    """ Convert Intensity I(r) at r to I at theta_0 with multi-power law.
        theata_s and r in pixel """
    i = np.digitize(r, theta_s_pix, right=True) - 1
    I0 = I * r**(n_s[i]) * theta_s_pix[0]**(-n_s[0])
    for j in range(i):
        I0 *= theta_s_pix[j+1]**(n_s[j]-n_s[j+1])
        
    return I0

def I02I_mpow(n_s, theta_s_pix, r, I0=1):
    """ Convert Intensity I(r) at r to I at theta_0 with multi-power law.
        theata_s and r in pixel """
    i = np.digitize(r, theta_s_pix, right=True) - 1
        
    I = I0 / r**(n_s[i]) / theta_s_pix[0]**(-n_s[0])
    for j in range(i):
        I *= theta_s_pix[j+1]**(n_s[j+1]-n_s[j])
        
    return I


def calculate_external_light_pow(n0, theta0, pos_source, pos_eval, I0_source):
    # Calculate light produced by source (I0, pos_source) at pos_eval. 
    r_s = distance.cdist(pos_source,  pos_eval)
    
    I0_s = np.repeat(I0_source[:, np.newaxis], r_s.shape[-1], axis=1) 
    
    I_s = I0_s / (r_s/theta0)**n0
    I_s[(r_s==0)] = 0
    
    return I_s.sum(axis=0)

def calculate_external_light_mpow(n_s, theta_s_pix, pos_source, pos_eval, I0_source):
    """ Calculate light produced by source (I0_source, pos_source) at pos_eval. """ 
    r_s = distance.cdist(pos_source, pos_eval)
    r_inds = np.digitize(r_s, theta_s_pix, right=True) - 1
    
    r_inds_uni, r_inds_inv = np.unique(r_inds, return_inverse=True)
    
    I0_s = np.repeat(I0_source[:, np.newaxis], r_s.shape[-1], axis=1) 
    
    I_s = I0_s * theta_s_pix[0]**n_s[0] / r_s**(n_s[r_inds])
    factors = np.array([np.prod([theta_s_pix[j+1]**(n_s[j+1]-n_s[j])
                                 for j in range(i)]) for i in r_inds_uni])
    I_s *= factors[r_inds_inv].reshape(len(I0_source),-1)
    
    I_s[(r_s==0)] = 0
    
    return I_s.sum(axis=0)


def I2Flux_mpow(frac, n_s, theta_s, r, I=1):
    """ Convert Intensity I(r) at r to total flux with fraction of multi-power law.
        theata_s and r in pixel """

    I0 = I2I0_mpow(n_s, theta_s, r, I=I)
    Flux_mpow = multi_power2d_Amp2Flux(n_s=n_s, theta_s=theta_s, Amp=I0)
    Flux_tot = Flux_mpow / frac
    
    return Flux_tot

def Flux2I_mpow(frac, n_s, theta_s, r, Flux=1):
    """ Convert total flux to Intensity I(r) at r.
        theata_s and r in pixel """
    i = np.digitize(r, theta_s, right=True) - 1
    
    Flux_mpow = Flux * frac
    I0 = multi_power2d_Flux2Amp(n_s=n_s, theta_s=theta_s, Flux=Flux_mpow)
    
    I = I0 / r**(n_s[i]) / theta_s[0]**(-n_s[0])
    for j in range(i):
        I /= theta_s[j+1]**(n_s[j]-n_s[j+1])

    return I


### 1D/2D conversion factor ###

def C_mof2Dto1D(r_core, beta):
    """ gamma in pixel """
    return 1./(beta-1) * 2*math.sqrt(np.pi) * r_core * Gamma(beta) / Gamma(beta-1./2) 

def C_mof1Dto2D(r_core, beta):
    """ gamma in pixel """
    return 1. / C_mof2Dto1D(r_core, beta)


def C_pow2Dto1D(n, theta0):
    """ theta0 in pixel """
    return np.pi * theta0 * (n-1) / (n-2)

def C_pow1Dto2D(n, theta0):
    """ theta0 in pixel """
    return 1. / C_pow2Dto1D(n, theta0)


def C_mpow2Dto1D(n_s, theta_s):
    """ theta in pixel """
    a_s = compute_multi_pow_norm(n_s, theta_s, 1)
    n0, theta0, a0 = n_s[0], theta_s[0], a_s[0]
 
    I_2D = 1. * np.pi * theta0**2
    for k in range(len(n_s)-1):
        if n_s[k] == 2:
            I_2D += 2*np.pi * a_s[k] * np.log(theta_s[k+1]/theta_s[k])
        else:
            I_2D += 2*np.pi * a_s[k] * (theta_s[k]**(2-n_s[k]) - theta_s[k+1]**(2-n_s[k])) / (n_s[k]-2) 
    I_2D += 2*np.pi * a_s[-1] * theta_s[-1]**(2-n_s[-1]) / (n_s[-1]-2)   
    
    I_1D = 1. * theta0
    for k in range(len(n_s)-1):
        if n_s[k] == 1:
            I_1D += a_s[k] * np.log(theta_s[k+1]/theta_s[k])
        else:
            I_1D += a_s[k] * (theta_s[k]**(1-n_s[k]) - theta_s[k+1]**(1-n_s[k])) / (n_s[k]-1) 
    I_1D += a_s[-1] * theta_s[-1]**(1-n_s[-1]) / (n_s[-1]-1)
    
    return I_2D / I_1D 


def C_mpow1Dto2D(n_s, theta_s):
    """ theta in pixel """
    return 1. / C_mpow2Dto1D(n_s, theta_s)

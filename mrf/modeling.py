#########################################################################
################### From Qing Liu (UToronto) ############################
#########################################################################
import numpy as np
import math
from astropy.io import fits
import galsim
from galsim import GalSimBoundsError

from numpy.polynomial.legendre import leggrid2d
from scipy.integrate import quad
from itertools import combinations
from functools import partial
from astropy.utils import lazyproperty
from copy import deepcopy

############################################
# Functions for making PSF models
############################################
from types import SimpleNamespace    

class PSF_Model:
    """ A PSF Model object """
    def __init__(self, params=None, core_model='moffat', aureole_model='power'):
        """
        Parameters
        ----------
        params : a dictionary containing keywords of PSF parameter
        core_model : model of PSF core (moffat)
        aureole_model : model of aureole ("power" or "multi-power")	
        
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
                
    def make_grid(self, image_size, pixel_scale=2.5):
        """ Build grid for drawing """
        self.image_size = image_size
        self.yy, self.xx = np.mgrid[:image_size, :image_size]
        self.cen = ((image_size-1)/2., (image_size-1)/2.)
        self.pixel_scale = pixel_scale
        
        for key, val in self.params.items():
            if (key == 'gamma') | ('theta' in key):
                val = val / pixel_scale
                exec('self.' + key + '_pix' + ' = val')
                
    def update(self, params):
        """ Update PSF parameters from dictionary keys """
        for key, val in params.items():
            if np.ndim(val) > 0:
                val = np.array(val)
                
            exec('self.' + key + ' = val')
            self.params[key] = val
            
            if 'theta' in key:
                val = val / self.pixel_scale
                exec('self.' + key + '_pix' + ' = val')
                
    def copy(self):
        """ A deep copy of the object """
        return deepcopy(self)            

    @property
    def f_core1D(self):
        """ 1D Core function """
        c_mof2Dto1D = C_mof2Dto1D(self.gamma_pix, self.beta)
        return lambda r: moffat1d_normed(r, self.gamma_pix, self.beta) / c_mof2Dto1D

    @property
    def f_aureole1D(self):
        """ 1D Aureole function """
        if self.aureole_model == "power":
            n, theta_0_pix = self.n, self.theta_0_pix
            c_aureole_2Dto1D = C_pow2Dto1D(n, theta_0_pix)
            f_aureole = lambda r: trunc_power1d_normed(r, n, theta_0_pix) / c_aureole_2Dto1D

        if self.aureole_model == "multi-power":
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
                
    def generate_core(self, folding_threshold=1.e-10):
        """ Generate Galsim PSF of core. """
        self.fwhm = self.gamma * 2. * np.sqrt(2**(1./self.beta)-1)
        gsparams = galsim.GSParams(folding_threshold=folding_threshold)
        psf_core = galsim.Moffat(beta=self.beta, fwhm=self.fwhm,
                                flux=1., gsparams=gsparams) # in arcsec
        self.psf_core = psf_core
        return psf_core
    
    def generate_aureole(self,
                         contrast=1e6,
                         psf_scale=None,
                         psf_range=None,
                         min_psf_range=30,
                         max_psf_range=600,
                         interpolant="cubic"):
        """
        Generate Galsim PSF of aureole.

        Parameters
        ----------
        contrast: Ratio of the intensity at max range and at center. Used to calculate the PSF size if not given.
        psf_scale: Pixel scale of the PSF, in general <= pixel scale of data. In arcsec/pix.
        psf_range: Range of PSF. In arcsec.
        min_psf_range : Minimum range of PSF. In arcsec.
        max_psf_range : Maximum range of PSF. In arcsec.
        interpolant: Interpolant method in Galsim.
        
        Returns
        ----------
        psf_aureole: power law Galsim PSF, flux normalized to be 1.
        psf_size: Size of PSF used. In pixel.
        
        """

        
        if self.aureole_model == "power":
            n = self.n
            theta_0 = self.theta_0
            
        elif self.aureole_model == "multi-power":
            n_s = self.n_s
            theta_s = self.theta_s
            n = n_s[0]
            theta_0 = theta_s[0]
            
            self.theta_0 = theta_0
            self.n = n
            
        if psf_range is None:
            a = theta_0**n
            opt_psf_range = int((contrast * a) ** (1./n))
            psf_range = max(min_psf_range, min(opt_psf_range, max_psf_range))
        
        if psf_scale is None:
            psf_scale = 0.8 * self.pixel_scale
            
        # full (image) PSF size in pixel
        psf_size = 2 * psf_range // psf_scale

        # Generate Grid of PSF and plot PSF model in real space onto it
        cen_psf = ((psf_size - 1) / 2., (psf_size - 1) / 2.)
        yy_psf, xx_psf = np.mgrid[:psf_size, :psf_size]
        
        if self.aureole_model == "power":
            theta_0_pix = theta_0 / psf_scale
            psf_model = trunc_power2d(xx_psf, yy_psf, n, theta_0_pix, I_theta0=1, cen=cen_psf) 
            
        elif self.aureole_model == "multi-power":
            theta_s_pix = theta_s / psf_scale
            psf_model =  multi_power2d(xx_psf, yy_psf, n_s, theta_s_pix, 1, cen=cen_psf) 

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
    
    
    def I2I0(self, I, r=10):
        """ Convert aureole I(r) at r to I0. r in pixel """
        
        if self.aureole_model == "power":
            return I2I0_pow(self.n, self.theta_0_pix, r, I=I)
        
        elif self.aureole_model == "multi-power":
            return I2I0_mpow(self.n_s, self.theta_s_pix, r, I=I)
        
    def I02I(self, I0, r=10):
        """ Convert aureole I(r) at r to I0. r in pixel """
        
        if self.aureole_model == "power":
            return I02I_pow(self.n, self.theta_0_pix, r, I0=I0)
        
        elif self.aureole_model == "multi-power":
            return I02I_mpow(self.n_s, self.theta_s_pix, r, I0=I0)
    
    def calculate_external_light(self, stars, n_iter=2):
        """ Calculate the integrated external scatter light that affects the flux scaling from very bright stars on the other stars.
        
        Parameters
        ----------
        stars : Star object
        n_iter : iteration time to do the calculation
        
        """
        I_ext = np.zeros(stars.n_bright)
        z_norm_verybright0 = stars.z_norm_verybright
        pos_source, pos_eval = stars.star_pos_verybright, stars.star_pos_bright
        
        if self.aureole_model == "power":
            cal_ext_light = partial(calculate_external_light_pow,
                                   n0=self.n, theta0=self.theta_s_pix,
                                   pos_source=pos_source, pos_eval=pos_eval)
        elif self.aureole_model == "multi-power":
            cal_ext_light = partial(calculate_external_light_mpow,
                                   n_s=self.n_s, theta_s=self.theta_s_pix,
                                   pos_source=pos_source, pos_eval=pos_eval)
            
        for i in range(n_iter):
            z_norm_verybright = z_norm_verybright0 - I_ext[:stars.n_verybright]
            I0_verybright = self.I2I0(z_norm_verybright, r=stars.r_scale)
            I_ext = cal_ext_light(I0=I0_verybright)
            
        return I_ext
    
    def I2Flux(self, I, r=10):
        """ Convert aureole I(r) at r to total flux. r in pixel """
        
        if self.aureole_model == "power":
            return I2Flux_pow(self.frac, self.n, self.theta_0_pix, r, I=I)
        
        elif self.aureole_model == "multi-power":
            return I2Flux_mpow(self.frac, self.n_s, self.theta_s_pix, r, I=I)
        
    def Flux2I(self, Flux, r=10):
        """ Convert aureole I(r) at r to total flux. r in pixel """
        
        if self.aureole_model == "power":
            return Flux2I_pow(self.frac, self.n, self.theta_0_pix, r, Flux=Flux)
        
        elif self.aureole_model == "multi-power":
            return Flux2I_mpow(self.frac, self.n_s, self.theta_s_pix, r,  Flux=Flux)
        
    def SB2Flux(self, SB, BKG, ZP, r=10):
        """
        Convert suface brightness SB at r to total flux, given background value and ZP. 
        """
        I = SB2Intensity(SB, BKG, ZP, self.pixel_scale) - BKG # Intensity = I + BKG
        return self.I2Flux(I, r=r)
    
    def Flux2SB(self, Flux, BKG, ZP, r=10):
        """ Convert total flux to suface brightness SB at r, given background value and ZP. """
        I = self.Flux2I(Flux, r=r)
        return Intensity2SB(I+ BKG, BKG, ZP, self.pixel_scale)
    
    @property
    def psf_star(self):
        """ Galsim object of star psf (core+aureole) """
        frac = self.frac
        psf_core, psf_aureole = self.psf_core, self.psf_aureole
        return (1 - frac) * psf_core + frac * psf_aureole
    

    def plot_PSF_model_galsim(self, contrast=None, save=False, dir_name='.'):
        """ Build and plot Galsim 2D model averaged in 1D """
        from .display import plot_PSF_model_galsim
        image_psf = plot_PSF_model_galsim(self, contrast=None,
                                          save=save, dir_name=dir_name)
        self.image_psf = image_psf
        return image_psf
    
    @staticmethod
    def write_psf_image(image_psf, filename='PSF_model.fits'):
        """ Write the 2D psf image to fits """
        hdu = fits.ImageHDU(image_psf)
        hdu.writeto(filename, overwrite=True)
    
    def draw_core2D_in_real(self, star_pos, Flux):
        """ 2D drawing function of the core in real space given positions and flux (of core) of target stars """
        from astropy.modeling.models import Moffat2D
        gamma, alpha = self.gamma_pix, self.beta
        Amps = np.array([moffat2d_Flux2Amp(gamma, alpha, Flux=flux)
                       for flux in Flux])
        f_core_2d_s = np.array([Moffat2D(amplitude=amp, x_0=x0, y_0=y0,
                                                gamma=gamma, alpha=alpha)
                                for ((x0,y0), amp) in zip(star_pos, Amps)])
            
        return f_core_2d_s
        
    def draw_aureole2D_in_real(self, star_pos, Flux=None, I0=None):
        """ 2D drawing function of the aureole in real space given positions and flux / amplitude (of aureole) of target stars """
        
        if self.aureole_model == "power":
            n = self.n
            theta_0_pix = self.theta_0_pix
            
            if I0 is None:
                I_theta0 = power2d_Flux2Amp(n, theta_0_pix, Flux=1) * Flux
            elif Flux is None:
                I_theta0 = I0
            else:
                raise MyError("Both Flux and I0 are not given.")
            
            f_aureole_2d_s = np.array([lambda xx, yy, cen=pos, I=I:\
                                      trunc_power2d(xx, yy, cen=cen,
                                                    n=n, theta0=theta_0_pix,
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

def coord_Im2Array(X_IMAGE, Y_IMAGE, origin=1):
    """ Convert image coordniate to numpy array coordinate """
    x_arr, y_arr = int(max(round(Y_IMAGE) - origin, 0)), int(max(round(X_IMAGE) - origin, 0))
    return x_arr, y_arr

def coord_Array2Im(x_arr, y_arr, origin=1):
    """ Convert image coordniate to numpy array coordinate """
    X_IMAGE, Y_IMAGE = y_arr + origin, x_arr + origin
    return X_IMAGE, Y_IMAGE

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

def cal_profile_1d(img, cen=None, mask=None, back=None, bins=None,
                   color="steelblue", xunit="pix", yunit="Intensity",
                   seeing=2.5, pixel_scale=2.5, ZP=27.1, sky_mean=884, sky_std=3, dr=1.5, 
                   lw=2, alpha=0.7, markersize=5, I_shift=0, figsize=None,
                   core_undersample=False, label=None, plot_line=False, mock=False,
                   plot=True, scatter=False, fill=False, errorbar=False, verbose=False):
    """Calculate 1d radial profile of a given star postage"""
    if mask is None:
        mask =  np.zeros_like(img, dtype=bool)
    if back is None:     
        back = np.ones_like(img) * sky_mean
    if cen is None:
        cen = (img.shape[0]-1)/2., (img.shape[1]-1)/2.
        
    yy, xx = np.indices((img.shape))
    rr = np.sqrt((xx - cen[0])**2 + (yy - cen[1])**2)
    r = rr[~mask].ravel()  # radius in pix
    z = img[~mask].ravel()  # pixel intensity
    r_core = np.int(3 * seeing/pixel_scale) # core radius in pix

    # Decide the outermost radial bin r_max before going into the background
    bkg_cumsum = np.arange(1, len(z)+1, 1) * np.median(back)
    z_diff =  abs(z.cumsum() - bkg_cumsum)
    n_pix_max = len(z) - np.argmin(abs(z_diff - 0.0001 * z_diff[-1]))
    r_max = np.sqrt(n_pix_max/np.pi)
    r_max = np.min([img.shape[0]//2, r_max])
    
    if verbose:
        print("Maximum R: %d (pix)"%np.int(r_max))    
    
    if xunit == "arcsec":
        r = r * pixel_scale   # radius in arcsec
        r_core = r_core * pixel_scale
        r_max = r_max * pixel_scale
        
    d_r = dr * pixel_scale if xunit == "arcsec" else dr
    
    if mock:
        clip = lambda z: sigma_clip((z), sigma=3, maxiters=5)
    else:
        clip = lambda z: 10**sigma_clip(np.log10(z+1e-10), sigma=3, maxiters=5)
        
    if bins is None:
        # Radial bins: discrete/linear within r_core + log beyond it
        if core_undersample:  
            # for undersampled core, bin in individual pixels 
            bins_inner = np.unique(r[r<r_core]) + 1e-3
        else: 
            bins_inner = np.linspace(0, r_core, int(min((r_core/d_r*2), 5))) + 1e-3

        n_bin_outer = np.max([7, np.min([np.int(r_max/d_r/10), 50])])
        if r_max > (r_core+d_r):
            bins_outer = np.logspace(np.log10(r_core+d_r), np.log10(r_max-d_r), n_bin_outer)
        else:
            bins_outer = []
        bins = np.concatenate([bins_inner, bins_outer])
        _, bins = np.histogram(r, bins=bins)
    
    # Calculate binned 1d profile
    r_rbin = np.array([])
    z_rbin = np.array([])
    zstd_rbin = np.array([])
    for k, b in enumerate(bins[:-1]):
        in_bin = (r>bins[k])&(r<bins[k+1])
        
        z_clip = clip(z[in_bin])
        if len(z_clip)==0:
            continue

        zb = np.mean(z_clip)
        zstd_b = np.std(z_clip)
        
        z_rbin = np.append(z_rbin, zb)
        zstd_rbin = np.append(zstd_rbin, zstd_b)
        r_rbin = np.append(r_rbin, np.mean(r[in_bin]))
        
        
    logzerr_rbin = 0.434 * abs( zstd_rbin / (z_rbin-sky_mean))
    
    if plot:
        if figsize is not None:
            plt.figure(figsize=figsize)
        if yunit == "Intensity":  
            # plot radius in Intensity
            plt.plot(r_rbin, np.log10(z_rbin), "-o", color=color,
                     mec="k", lw=lw, ms=markersize, alpha=alpha, zorder=3, label=label) 
            if scatter:
                I = np.log10(z)
                
            if fill:
                plt.fill_between(r_rbin, np.log10(z_rbin)-logzerr_rbin, np.log10(z_rbin)+logzerr_rbin,
                                 color=color, alpha=0.2, zorder=1)
            plt.ylabel("log Intensity")
            plt.xscale("log")
            plt.xlim(r_rbin[np.isfinite(r_rbin)][0]*0.8, r_rbin[np.isfinite(r_rbin)][-1]*1.2)

        elif yunit == "SB":  
            # plot radius in Surface Brightness
            I_rbin = Intensity2SB(I=z_rbin, BKG=np.median(back),
                                  ZP=ZP, pixel_scale=pixel_scale) + I_shift
            I_sky = -2.5*np.log10(sky_std) + ZP + 2.5 * math.log10(pixel_scale**2)

            plt.plot(r_rbin, I_rbin, "-o", mec="k",
                     lw=lw, ms=markersize, color=color, alpha=alpha, zorder=3, label=label)   
            if scatter:
                I = Intensity2SB(I=z, BKG=np.median(back),
                                 ZP=ZP, pixel_scale=pixel_scale) + I_shift
                
            if errorbar:
                Ierr_rbin_up = I_rbin - Intensity2SB(I=z_rbin,
                                                     BKG=np.median(back)-sky_std,
                                                     ZP=ZP, pixel_scale=pixel_scale)
                Ierr_rbin_lo = Intensity2SB(I=z_rbin-sky_std,
                                            BKG=np.median(back)+sky_std,
                                            ZP=ZP, pixel_scale=pixel_scale) - I_rbin
                lolims = np.isnan(Ierr_rbin_lo)
                uplims = np.isnan(Ierr_rbin_up)
                Ierr_rbin_lo[lolims] = 4
                Ierr_rbin_up[uplims] = 4
                plt.errorbar(r_rbin, I_rbin, yerr=[Ierr_rbin_up, Ierr_rbin_lo],
                             fmt='', ecolor=color, capsize=2, alpha=0.5)
                
            plt.ylabel("Surface Brightness [mag/arcsec$^2$]")        
            plt.gca().invert_yaxis()
            plt.xscale("log")
            plt.xlim(r_rbin[np.isfinite(r_rbin)][0]*0.8,r_rbin[np.isfinite(r_rbin)][-1]*1.2)
            plt.ylim(30,17)
        
        if scatter:
            plt.scatter(r[r<3*r_core], I[r<3*r_core], color=color, 
                        s=markersize/2, alpha=0.2, zorder=1)
            plt.scatter(r[r>=3*r_core], I[r>=3*r_core], color=color,
                        s=markersize/4, alpha=0.1, zorder=1)
            
        plt.xlabel("r [acrsec]") if xunit == "arcsec" else plt.xlabel("r [pix]")

        # Decide the radius within which the intensity saturated for bright stars w/ intersity drop half
        dz_rbin = np.diff(np.log10(z_rbin)) 
        dz_cum = np.cumsum(dz_rbin)

        if plot_line:
            r_satr = r_rbin[np.argmax(dz_cum<-0.3)] + 1e-3
            plt.axvline(r_satr,color="k",ls="--",alpha=0.9)
            plt.axvline(r_core,color="k",ls=":",alpha=0.9)
            plt.axhline(I_sky,color="gray",ls="-.",alpha=0.7)
        
    if yunit == "Intensity":
        return r_rbin, z_rbin, logzerr_rbin
    elif yunit == "SB": 
        return r_rbin, I_rbin, None    


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

def multi_power2d_cover(xx, yy, n0, theta0, I_theta0, n_s, theta_s, cen):
    rr = np.sqrt((xx-cen[0])**2 + (yy-cen[1])**2) + 1e-6
    a0 = I_theta0/(theta0)**(-n0)
    z = a0 * np.power(rr, -n0) 
    z[rr<=theta0] = I_theta0
    
    I_theta_i = a0 * float(theta_s[0])**(-n0)
    
    for i, (n_i, theta_i) in enumerate(zip(n_s, theta_s)):
        a_i = I_theta_i/(theta_i)**(-n_i)
        z_i = a_i * np.power(rr, -n_i)
        z[rr>theta_i] = z_i[rr>theta_i]
        try:
            I_theta_i = a_i * float(theta_s[i+1])**(-n_i)
        except IndexError:
            pass
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

def compute_multi_pow_norm0(n0, n_s, theta0, theta_s, I_theta0):
    """ Compute normalization factor of each power law component """
    a_s = np.zeros(len(n_s))
    a0 = I_theta0 * theta0**(n0)

    I_theta_i = a0 * float(theta_s[0])**(-n0)
    for i, (n_i, theta_i) in enumerate(zip(n_s, theta_s)):
        a_i = I_theta_i/(theta_i)**(-n_i)
        try:
            a_s[i] = a_i
            I_theta_i = a_i * float(theta_s[i+1])**(-n_i)
        except IndexError:
            pass    
    return a0, a_s

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

def map2d(f, xx=None, yy=None):
    return f(xx,yy)

def map2d_k(k, func_list, xx=None, yy=None):
    return func_list[k](xx, yy)


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


### 1D/2D conversion factor ###

def C_mof2Dto1D(r_core, beta):
    """ gamma in pixel """
    return 1. / (beta-1) * 2 * np.sqrt(np.pi) * r_core * math.gamma(beta) / math.gamma(beta - 1. / 2) 

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



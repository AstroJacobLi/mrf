import os
import sys
import gc
import copy
import yaml
import logging

import numpy as np
import matplotlib.pyplot as plt

from astropy import wcs
from astropy.io import fits
from astropy.table import Table, Column, hstack, vstack

import warnings
warnings.filterwarnings("ignore")

class Config(object):
    """
    Configuration class.
    """
    def __init__(self, d):
        for a, b in d.items():
            if isinstance(b, (list, tuple)):
                setattr(self, a, [Config(x) if isinstance(x, dict) else x for x in b])
            else:
                setattr(self, a, Config(b) if isinstance(b, dict) else b)
        Config.config = d
    
    def complete_config(self):
        """
        This function fill vacant parameters in the config file with default values. 

        Parameters:
            config (Config class): configuration of the MRF task.
        Returns:
            config (Config class)
        """
        # sex
        default_sex = {
            'b': 64,
            'f': 3,
            'sigma': 3.0,
            'minarea': 2,
            'deblend_cont': 0.005,
            'deblend_nthresh': 32,
            'sky_subtract': True,
            'flux_aper': [3, 6],
            'show_fig': False
        }
        for name in default_sex.keys():
            if not name in self.sex.__dict__.keys():
                setattr(self.sex, name, default_sex[name])
        
        # fluxmodel
        default = {'gaussian_radius': 1.5,
                'gaussian_threshold': 0.05,
                'unmask_lowsb': False,
                'sb_lim': 26.0,
                'unmask_ratio': 3,
                'interp': 'iraf',
                'minarea': 25
                }
        for name in default.keys():
            if not name in self.fluxmodel.__dict__.keys():
                setattr(self.fluxmodel, name, default[name])

        # kernel
        default = {
            'kernel_size': 8,
            'kernel_edge': 1,
            'nkernel': 25,
            'circularize': False,
            'show_fig': True
        }
        for name in default.keys():
            if not name in self.kernel.__dict__.keys():
                setattr(self.kernel, name, default[name])
        
        # starhalo
        default = {
            'fwhm_lim': 200,
            'padsize': 50,
            'edgesize': 5,
            'b': 32,
            'f': 3,
            'sigma': 3.5,
            'minarea': 3,
            'deblend_cont': 0.003,
            'deblend_nthresh': 32,
            'sky_subtract': True,
            'flux_aper': [3, 6],
            'mask_contam': True,
            'interp': 'iraf',
            'cval': 'nan'
        }
        for name in default.keys():
            if not name in self.starhalo.__dict__.keys():
                setattr(self.starhalo, name, default[name])

        # Clean
        default = {
            'clean_img': True,
            'clean_file': False,
            'replace_with_noise': False,
            'gaussian_radius': 1.5,
            'gaussian_threshold': 0.003,
            'bright_lim': 16.5,
            'r': 8.0
        }
        for name in default.keys():
            if not name in self.clean.__dict__.keys():
                setattr(self.clean, name, default[name])

class Results():
    """
    Results class. Other attributes will be added by ``setattr()``.
    """
    def __init__(self, config):
        self.config = config
    
class MrfTask():
    '''
    MRF task class. This class implements `mrf`, with wide-angle PSF incorporated.
    '''
    def __init__(self, config_file):
        """
        Initialize ``MrfTask`` class. 

        Parameters:
            config_file (str): the directory of configuration YAML file.
        Returns:
            None
        """
        # Open configuration file
        with open(config_file, 'r') as ymlfile:
            cfg = yaml.safe_load(ymlfile)
            config = Config(cfg)
            config.complete_config()
        self.config_file = config_file
        self.config = config

    def set_logger(self, output_name='mrf', verbose=True):
        """
        Set logger for ``MrfTask``. The logger will record the time and each output. The log file will be saved locally.

        Parameters:
            verbose (bool): If False, the logger will be silent. 

        Returns:
            logger (``logging.logger`` object)
        """
        if verbose:
            log_filename = output_name + '.log'
            logging.basicConfig(format='%(asctime)s %(message)s', level=logging.INFO, 
                                handlers=[logging.StreamHandler(sys.stdout),
                                          logging.FileHandler(log_filename, mode='w')])
            self.logger = logging.getLogger(log_filename)                          
        else:
            logger = logging.getLogger('mylogger')
            logger.propagate = False
            self.logger = logger
        return self.logger

    def run(self, dir_lowres, dir_hires_b, dir_hires_r, certain_gal_cat, 
            wide_psf=True, output_name='mrf', verbose=True, skip_resize=False, 
            skip_SE=False, skip_mast=False):
        """
        Run MRF task.

        Parameters:
            dir_lowres (string): directory of input low-resolution image.
            dir_hires_b (string): directory of input high-resolution 
                blue-band image (typically g-band).
            dir_hires_r (string): directory of input high-resolution 
                red-band image (typically r-band).
            certain_gal_cat (string): directory of a catalog (in ascii format) which contains 
                RA and DEC of galaxies which you want to retain during MRF.
            wide_psf (bool): whether subtract bright stars using the wide-PSF of **Dragonfly**. 
                See Q. Liu et al. (in prep.) for details. 
            output_name (string): which will be the prefix of output files.
            verbose (bool): If True, it will make a log file recording the process. 
            skip_resize (bool): If True, the code will not `zoom` the images again but 
                use the resized images under the current directory. 
                This is designed for the case when you need to tweak parameters.
            skip_SE (bool): If True, the code will not repeat running SExtractor on two-bands high-res images, 
                but use the existing flux model under the current directory. 
                This is designed for the case when you need to tweak parameters.
            skip_mast (bool): Just for ``wide_psf=True`` mode. If True, the code will not 
                repeat retrieving Pan-STARRS catalog from MAST server, 
                but use the existing catalog under the current directory. 
                This is designed for the case when you need to tweak parameters.

        Returns:
            results (`Results` class): containing key results of this task.
        
        """
        from astropy.coordinates import SkyCoord, match_coordinates_sky
        from astropy.convolution import convolve, Box2DKernel, Gaussian2DKernel
        import astropy.units as u
        from mrf.utils import (save_to_fits, Flux_Model, mask_out_stars, extract_obj, \
                            bright_star_mask, Autokernel, psf_bkgsub)
        from mrf.utils import seg_remove_obj, mask_out_certain_galaxy

        from mrf.display import display_single, SEG_CMAP, display_multiple, draw_circles
        from mrf.celestial import Celestial, Star
        from mrf.utils import Config
        from reproject import reproject_interp, reproject_exact

        config = self.config
        logger = self.set_logger(output_name=output_name, verbose=verbose)
        results = Results(config)
        
        assert (
            ((config.lowres.dataset.lower() != 'df' or config.lowres.dataset.lower() != 'dragonfly') and wide_psf == True) or wide_psf == False
            ), "Wide PSF subtraction is only available for Dragonfly data. Check your low-resolution images!"

        logger.info('Running Multi-Resolution Filtering (MRF) on "{0}" and "{1}" images!'.format(config.hires.dataset, config.lowres.dataset))
        setattr(results, 'lowres_name', config.lowres.dataset)
        setattr(results, 'hires_name', config.hires.dataset)
        setattr(results, 'output_name', output_name)
        
        # 1. subtract background of lowres, if desired
        assert isinstance(dir_lowres, str), 'Input "img_lowres" must be string!'
        hdu = fits.open(dir_lowres)
        lowres = Celestial(hdu[0].data, header=hdu[0].header)
        if config.lowres.sub_bkgval:
            logger.info('Subtract BACKVAL=%.1f of Dragonfly image', float(lowres.header['BACKVAL']))
            lowres.image -= float(lowres.header['BACKVAL'])
        hdu.close()
        setattr(results, 'lowres_input', copy.deepcopy(lowres))
        
        # 2. Create magnified low-res image, and register high-res images with subsampled low-res ones
        f_magnify = config.lowres.magnify_factor
        logger.info('Magnify Dragonfly image with a factor of %.1f:', f_magnify)
        if skip_resize:
            hdu = fits.open('_lowres_{}.fits'.format(int(f_magnify)))
            lowres = Celestial(hdu[0].data, header=hdu[0].header)
        else:
            lowres.resize_image(f_magnify, method=config.fluxmodel.interp)
            lowres.save_to_fits('_lowres_{}.fits'.format(int(f_magnify)))

        logger.info('Register high resolution image "{0}" with "{1}"'.format(dir_hires_b, dir_lowres))
        if skip_resize:
            hdu = fits.open('_hires_b_reproj.fits')
            hires_b = Celestial(hdu[0].data, header=hdu[0].header)
            hdu.close()
            hdu = fits.open('_hires_r_reproj.fits')
            hires_r = Celestial(hdu[0].data, header=hdu[0].header)
            hdu.close()
        else:
            hdu = fits.open(dir_hires_b)
            if 'hsc' in dir_hires_b:
                array, _ = reproject_interp(hdu[1], lowres.header)
                # Note that reproject_interp does not conserve total flux
                # A factor is needed for correction.
                factor = (lowres.pixel_scale / (hdu[1].header['CD2_2'] * 3600))**2
                array *= factor
            else:
                array, _ = reproject_interp(hdu[0], lowres.header)
                factor = (lowres.pixel_scale / (hdu[0].header['CD2_2'] * 3600))**2
                array *= factor
            
            hires_b = Celestial(array, header=lowres.header)
            hires_b.save_to_fits('_hires_b_reproj.fits')
            hdu.close()
            
            logger.info('Register high resolution image "{0}" with "{1}"'.format(dir_hires_r, dir_lowres))
            hdu = fits.open(dir_hires_r)
            if 'hsc' in dir_hires_r:
                array, _ = reproject_interp(hdu[1], lowres.header)
                factor = (lowres.pixel_scale / (hdu[1].header['CD2_2'] * 3600))**2
                array *= factor
            else:
                array, _ = reproject_interp(hdu[0], lowres.header)
                factor = (lowres.pixel_scale / (hdu[0].header['CD2_2'] * 3600))**2
                array *= factor
            hires_r = Celestial(array, header=lowres.header)
            hires_r.save_to_fits('_hires_r_reproj.fits')
            hdu.close()

        # 3. Extract sources on hires images using SEP
        sigma = config.sex.sigma
        minarea = config.sex.minarea
        b = config.sex.b
        f = config.sex.f
        deblend_cont = config.sex.deblend_cont
        deblend_nthresh = config.sex.deblend_nthresh
        sky_subtract = config.sex.sky_subtract
        flux_aper = config.sex.flux_aper
        show_fig = config.sex.show_fig
            
        if skip_SE:
            hdu = fits.open('_hires_{}.fits'.format(int(f_magnify)))
            hires_3 = Celestial(hdu[0].data, header=hdu[0].header)
            hdu.close()
            hdu = fits.open('_colratio.fits')
            col_ratio = hdu[0].data
            hdu.close()
        else:
            logger.info('Build flux models on high-resolution images: Blue band')
            logger.info('    - sigma = %.1f, minarea = %d', sigma, minarea)
            logger.info('    - deblend_cont = %.5f, deblend_nthres = %.1f', deblend_cont, deblend_nthresh)
            _, _, b_imflux = Flux_Model(hires_b.image, hires_b.header, sigma=sigma, minarea=minarea, 
                                        deblend_cont=deblend_cont, deblend_nthresh=deblend_nthresh, 
                                        sky_subtract=sky_subtract, save=True, logger=logger)
            
            logger.info('Build flux models on high-resolution images: Red band')
            logger.info('    - sigma = %.1f, minarea = %d', sigma, minarea)
            logger.info('    - deblend_cont = %.5f, deblend_nthres = %.1f', deblend_cont, deblend_nthresh)
            _, _, r_imflux = Flux_Model(hires_r.image, hires_b.header, sigma=sigma, minarea=minarea, 
                                        deblend_cont=deblend_cont, deblend_nthresh=deblend_nthresh, 
                                        sky_subtract=sky_subtract, save=True, logger=logger)
            

            # 4. Make color correction, remove artifacts as well
            logger.info('Make color correction to blue band, remove artifacts as well')
            col_ratio = (b_imflux / r_imflux)
            col_ratio[np.isnan(col_ratio) | np.isinf(col_ratio)] = 0 # remove artifacts
            save_to_fits(col_ratio, '_colratio.fits', header=hires_b.header)
            
            color_term = config.lowres.color_term
            logger.info('    - color_term = {}'.format(color_term))
            median_col = np.nanmedian(col_ratio[col_ratio != 0])
            logger.info('    - median_color (blue/red) = {:.5f}'.format(median_col))

            fluxratio = col_ratio / median_col
            fluxratio[(fluxratio < 0.1) | (fluxratio > 10)] = 1 # remove extreme values
            col_correct = np.power(fluxratio, color_term)
            save_to_fits(col_correct, '_colcorrect.fits', header=hires_b.header)

            if config.lowres.band == 'r':
                hires_3 = Celestial(hires_r.image * col_correct, header=hires_r.header)
            elif config.lowres.band == 'g':
                hires_3 = Celestial(hires_b.image * col_correct, header=hires_b.header)
            else:
                raise ValueError('config.lowres.band must be "g" or "r"!')
            
            _ = hires_3.save_to_fits('_hires_{}.fits'.format(int(f_magnify)))
            
            setattr(results, 'hires_img', copy.deepcopy(hires_3))

            # Clear memory
            del r_imflux, b_imflux, hires_b, hires_r
            gc.collect()

        # 5. Extract sources on hi-res corrected image
        logger.info('Extract objects from color-corrected high resolution image with:')
        logger.info('    - sigma = %.1f, minarea = %d', sigma, minarea)
        logger.info('    - deblend_cont = %.5f, deblend_nthres = %.1f', deblend_cont, deblend_nthresh)
        objects, segmap = extract_obj(hires_3.image, b=b, f=f, sigma=sigma, minarea=minarea, 
                                      show_fig=False, flux_aper=flux_aper, sky_subtract=sky_subtract,
                                      deblend_nthresh=deblend_nthresh, 
                                      deblend_cont=deblend_cont, logger=logger)
        objects.write('_hires_obj_cat.fits', format='fits', overwrite=True)
        
        # 6. Remove bright stars and certain galaxies
        logger.info('Remove bright stars from this segmentation map, using SEP results.')
        logger.info('    - Bright star limit = {}'.format(config.starhalo.bright_lim))
        seg = copy.deepcopy(segmap)
        mag = config.hires.zeropoint - 2.5 * np.log10(abs(objects['flux']))
        objects.add_column(Column(data=mag, name='mag'))
        flag = np.where(mag < config.starhalo.bright_lim)
        for obj in objects[flag]:
            seg = seg_remove_obj(seg, obj['x'], obj['y'])
        objects[flag].write('_bright_stars_3.fits', format='fits', overwrite=True)
        logger.info('    - {} stars removed. '.format(len(flag[0])))
        if certain_gal_cat is not None:
            # Mask out certain galaxy here.
            logger.info('Remove objects from catalog "{}"'.format(certain_gal_cat))
            gal_cat = Table.read(certain_gal_cat, format='ascii')
            seg = mask_out_certain_galaxy(seg, hires_3.header, gal_cat=gal_cat, logger=logger)
        save_to_fits(seg, '_seg_3.fits', header=hires_3.header)
        
        setattr(results, 'segmap_nostar_nogal', seg)

        # 7. Remove artifacts from `hires_3` by color ratio and then smooth it
        # multiply by mask created from ratio of images - this removes all objects that are
        # only in g or r but not in both (artifacts, transients, etc)
        mask = seg * (col_ratio != 0)
        mask[mask != 0] = 1
        # Then blow mask up
        from astropy.convolution import Gaussian2DKernel, Box2DKernel, convolve
        smooth_radius = config.fluxmodel.gaussian_radius
        mask_conv = copy.deepcopy(mask)
        mask_conv[mask_conv > 0] = 1
        mask_conv = convolve(mask_conv.astype(float), Gaussian2DKernel(smooth_radius))
        seg_mask = (mask_conv >= config.fluxmodel.gaussian_threshold)

        hires_fluxmod = Celestial(seg_mask * hires_3.image, header=hires_3.header)
        hires_fluxmod.image[np.isnan(hires_fluxmod.image)] = 0
        _ = hires_fluxmod.save_to_fits('_hires_fluxmod.fits')
        logger.info('Flux model from high resolution image has been built.')
        setattr(results, 'hires_fluxmod', hires_fluxmod)
        
        # 8. Build kernel based on some stars
        img_hires = Celestial(hires_3.image.byteswap().newbyteorder(), 
                              header=hires_3.header, dataset=config.hires.dataset)
        img_lowres = Celestial(lowres.image.byteswap().newbyteorder(), 
                              header=lowres.header, dataset=config.hires.dataset)

        logger.info('Build convolving kernel to degrade high resolution image.')
        kernel_med, good_cat = Autokernel(img_hires, img_lowres, 
                                        int(f_magnify * config.kernel.kernel_size), 
                                        int(f_magnify * (config.kernel.kernel_size - config.kernel.kernel_edge)), 
                                        frac_maxflux=config.kernel.frac_maxflux, 
                                        show_figure=config.kernel.show_fig,
                                        nkernels=config.kernel.nkernel, logger=logger)
        # You can also circularize the kernel
        if config.kernel.circularize:
            logger.info('Circularize the kernel.')
            from compsub.utils import circularize
            kernel_med = circularize(kernel_med, n=14)
        save_to_fits(kernel_med, '_kernel_median.fits')
        setattr(results, 'kernel_med', kernel_med)

        # 9. Convolve this kernel to high-res image
        from astropy.convolution import convolve_fft
        logger.info('    - Convolving image, this could be a bit slow @_@')
        conv_model = convolve_fft(hires_fluxmod.image, kernel_med, boundary='fill', 
                            fill_value=0, nan_treatment='fill', normalize_kernel=False, allow_huge=True)
        save_to_fits(conv_model, '_lowres_model_{}.fits'.format(int(f_magnify)), header=hires_3.header)
        
        # Optionally remove low surface brightness objects from model: 
        if config.fluxmodel.unmask_lowsb:
            logger.info('    - Removing low-SB objects (SB > {}) from flux model.'.format(config.fluxmodel.sb_lim))
            from .utils import remove_lowsb
            hires_flxmd = remove_lowsb(hires_fluxmod.image, conv_model, kernel_med, seg, 
                                        "_hires_obj_cat.fits", 
                                        SB_lim=config.fluxmodel.sb_lim, 
                                        zeropoint=config.hires.zeropoint, 
                                        pixel_size=hires_fluxmod.pixel_scale, 
                                        unmask_ratio=config.fluxmodel.unmask_ratio, 
                                        minarea=config.fluxmodel.minarea * f_magnify**2,
                                        gaussian_radius=config.fluxmodel.gaussian_radius, 
                                        gaussian_threshold=config.fluxmodel.gaussian_threshold, 
                                        header=hires_fluxmod.header, 
                                        logger=logger)

            logger.info('    - Convolving image, this could be a bit slow @_@')
            conv_model = convolve_fft(hires_flxmd, kernel_med, boundary='fill', 
                                 fill_value=0, nan_treatment='fill', normalize_kernel=False, allow_huge=True)
            save_to_fits(conv_model, '_lowres_model_clean_{}.fits'.format(f_magnify), header=hires_3.header)
            setattr(results, 'hires_fluxmod', hires_flxmd)

        lowres_model = Celestial(conv_model, header=hires_3.header)
        res = Celestial(lowres.image - lowres_model.image, header=lowres.header)
        res.save_to_fits('_res_{}.fits'.format(f_magnify))

        lowres_model.resize_image(1 / f_magnify, method=config.fluxmodel.interp)
        lowres_model.save_to_fits('_lowres_model.fits')
        setattr(results, 'lowres_model_compact', copy.deepcopy(lowres_model))

        res.resize_image(1 / f_magnify, method=config.fluxmodel.interp)
        res.save_to_fits(output_name + '_res.fits')
        setattr(results, 'res', res)
        logger.info('Compact objects has been subtracted from low-resolution image! Saved as "{}".'.format(output_name + '_res.fits'))

        # 10. Subtract bright star halos! Only for those left out in flux model!
        star_cat = Table.read('_bright_stars_3.fits', format='fits')
        star_cat['x'] /= f_magnify
        star_cat['y'] /= f_magnify
        ra, dec = res.wcs.wcs_pix2world(star_cat['x'], star_cat['y'], 0)
        star_cat.add_columns([Column(data=ra, name='ra'), Column(data=dec, name='dec')])

        b = config.starhalo.b
        f = config.starhalo.f
        sigma = config.starhalo.sigma
        minarea = config.starhalo.minarea
        deblend_cont = config.starhalo.deblend_cont
        deblend_nthresh = config.starhalo.deblend_nthresh
        sky_subtract = config.starhalo.sky_subtract
        flux_aper = config.starhalo.flux_aper

        logger.info('Extract objects from compact-object-subtracted low-resolution image with:')
        logger.info('    - sigma = %.1f, minarea = %d', sigma, minarea)
        logger.info('    - deblend_cont = %.5f, deblend_nthres = %.1f', deblend_cont, deblend_nthresh)
        objects, segmap = extract_obj(res.image, 
                                    b=b, f=f, sigma=sigma, minarea=minarea,
                                    deblend_nthresh=deblend_nthresh, 
                                    deblend_cont=deblend_cont, 
                                    sky_subtract=sky_subtract, show_fig=False, 
                                    flux_aper=flux_aper, logger=logger)
        
        ra, dec = res.wcs.wcs_pix2world(objects['x'], objects['y'], 0)
        objects.add_columns([Column(data=ra, name='ra'), Column(data=dec, name='dec')])
        
        # Match two catalogs
        logger.info('Stack stars to get PSF model!')
        logger.info('    - Match detected objects with previously discard stars')
        temp, sep2d, _ = match_coordinates_sky(SkyCoord(ra=star_cat['ra'], dec=star_cat['dec'], unit='deg'),
                                               SkyCoord(ra=objects['ra'], dec=objects['dec'], unit='deg'))
        #temp = temp[sep2d < 5 * u.arcsec]
        bright_star_cat = objects[np.unique(temp)]
        mag = config.lowres.zeropoint - 2.5 * np.log10(bright_star_cat['flux'])
        bright_star_cat.add_column(Column(data=mag, name='mag'))
        
        if certain_gal_cat is not None:
            ## Remove objects in GAL_CAT
            temp, dist, _ = match_coordinates_sky(
                                SkyCoord(ra=gal_cat['ra'], dec=gal_cat['dec'], unit='deg'),
                                SkyCoord(ra=bright_star_cat['ra'], dec=bright_star_cat['dec'], unit='deg'))
            to_remove = []
            for i, obj in enumerate(dist):
                if obj < 10 * u.arcsec:
                    to_remove.append(temp[i])
            if len(to_remove) != 0:
                bright_star_cat.remove_rows(np.unique(to_remove))
        
        bright_star_cat.write('_bright_star_cat.fits', format='fits', overwrite=True)
        setattr(results, 'bright_star_cat', bright_star_cat)

        #### Select non-edge good stars to stack ###
        halosize = config.starhalo.halosize
        padsize = config.starhalo.padsize
        # FWHM selection
        psf_cat = bright_star_cat[bright_star_cat['fwhm_custom'] < config.starhalo.fwhm_lim]
        # Mag selection
        psf_cat = psf_cat[psf_cat['mag'] < config.starhalo.bright_lim]
        psf_cat = psf_cat[psf_cat['mag'] > 12.0] # Discard heavily saturated stars

        ny, nx = res.image.shape
        non_edge_flag = np.logical_and.reduce([(psf_cat['x'] > padsize), (psf_cat['x'] < nx - padsize), 
                                               (psf_cat['y'] > padsize), (psf_cat['y'] < ny - padsize)])
        psf_cat = psf_cat[non_edge_flag]                                        
        psf_cat.sort('flux')
        psf_cat.reverse()
        psf_cat = psf_cat[:int(config.starhalo.n_stack)]
        logger.info('    - Get {} stars to be stacked!'.format(len(psf_cat)))
        setattr(results, 'psf_cat', psf_cat)

        # Construct and stack `Stars`.
        size = 2 * halosize + 1
        stack_set = np.zeros((len(psf_cat), size, size))
        bad_indices = []
        for i, obj in enumerate(psf_cat):
            try:
                sstar = Star(results.lowres_input.image, header=results.lowres_input.header, starobj=obj, 
                             halosize=halosize, padsize=padsize)
                cval = config.starhalo.cval
                if isinstance(cval, str) and 'nan' in cval.lower():
                    cval = np.nan
                else:
                    cval = float(cval)

                sstar.centralize(method=config.starhalo.interp)
                
                if config.starhalo.mask_contam is True:
                    sstar.mask_out_contam(sigma=4.0, deblend_cont=0.0001, show_fig=False, verbose=False)
                    #sstar.image = sstar.get_masked_image(cval=cval)
                    #sstar.mask_out_contam(sigma=3, deblend_cont=0.0001, show_fig=False, verbose=False)
                #sstar.sub_bkg(verbose=False)
                if config.starhalo.norm == 'flux_ann':
                    stack_set[i, :, :] = sstar.get_masked_image(cval=cval) / sstar.fluxann
                else:
                    stack_set[i, :, :] = sstar.get_masked_image(cval=cval) / sstar.flux
                
            except Exception as e:
                stack_set[i, :, :] = np.ones((size, size)) * 1e9
                bad_indices.append(i)
                logger.info(e)
                print(e)

        from astropy.stats import sigma_clip
        stack_set = np.delete(stack_set, bad_indices, axis=0)
        median_psf = np.nanmedian(stack_set, axis=0)
        median_psf = psf_bkgsub(median_psf, int(config.starhalo.edgesize))
        median_psf = convolve(median_psf, Box2DKernel(1))
        sclip = sigma_clip(stack_set, axis=0, maxiters=3)
        sclip.data[sclip.mask] = np.nan
        error_psf = np.nanstd(sclip.data, ddof=2, axis=0) / np.sqrt(np.sum(~np.isnan(sclip.data), axis=0))
        save_to_fits(median_psf, '_median_psf.fits');
        save_to_fits(error_psf, '_error_psf.fits');
        
        setattr(results, 'PSF', median_psf)
        setattr(results, 'PSF_err', error_psf)
        
        logger.info('    - Stars are stacked successfully!')
        save_to_fits(stack_set, '_stack_bright_stars.fits')
        
        # 11. Build starhalo models and then subtract from "res" image
        if wide_psf:
            results = self._subtract_widePSF(results, res, halosize, bright_star_cat, median_psf, lowres_model, output_name, skip_mast=skip_mast)
        else:
            results = self._subtract_stackedPSF(results, res, halosize, bright_star_cat, median_psf, lowres_model, output_name)
        
        img_sub = results.lowres_final_unmask.image

        # 12. Mask out dirty things!
        if config.clean.clean_img:
            logger.info('Clean the image!')
            model_mask = convolve(1e3 * results.lowres_model.image / np.nansum(results.lowres_model.image),
                                  Gaussian2DKernel(config.clean.gaussian_radius))
            model_mask[model_mask < config.clean.gaussian_threshold] = 0
            model_mask[model_mask != 0] = 1
            # Mask out very bright stars, according to their radius
            totmask = bright_star_mask(model_mask.astype(bool), bright_star_cat, 
                                       bright_lim=config.clean.bright_lim, 
                                       r=config.clean.r)
            totmask = convolve(totmask.astype(float), Box2DKernel(2))
            totmask[totmask > 0] = 1
            if config.clean.replace_with_noise:
                logger.info('    - Replace artifacts with noise.')
                from mrf.utils import img_replace_with_noise
                final_image = img_replace_with_noise(img_sub.byteswap().newbyteorder(), totmask)
            else:
                logger.info('    - Replace artifacts with void.')
                final_image = img_sub * (~totmask.astype(bool))
            
            save_to_fits(final_image, output_name + '_final.fits', header=res.header)
            save_to_fits(totmask.astype(float), output_name + '_mask.fits', header=res.header)
            setattr(results, 'lowres_final', Celestial(final_image, header=res.header))
            setattr(results, 'lowres_mask', Celestial(totmask.astype(float), header=res.header))
            logger.info('The final result is saved as "{}"!'.format(output_name + '_final.fits'))
            logger.info('The mask is saved as "{}"!'.format(output_name + '_mask.fits'))
        # Delete temp files
        if config.clean.clean_file:
            logger.info('Delete all temporary files!')
            os.system('rm -rf _*.fits')

        # 13. determine detection depth
        from .sbcontrast import cal_sbcontrast
        _  = cal_sbcontrast(final_image, totmask.astype(int), 
                             config.lowres.pixel_scale, config.lowres.zeropoint, 
                             scale_arcsec=60, minfrac=0.8, minback=6, verbose=True, logger=logger);
        
        # Plot out the result
        plt.rcParams['text.usetex'] = False
        fig, [ax1, ax2, ax3] = plt.subplots(1, 3, figsize=(15, 8))
        hdu = fits.open(dir_lowres)
        lowres_image = hdu[0].data
        ax1 = display_single(lowres_image, ax=ax1, scale_bar_length=300, 
                            scale_bar_y_offset=0.3, pixel_scale=config.lowres.pixel_scale, 
                            add_text='Lowres', text_y_offset=0.7)
        ax2 = display_single(lowres_model.image, ax=ax2, scale_bar=False, 
                            add_text='Model', text_y_offset=0.7)
        ax3 = display_single(final_image, ax=ax3, scale_bar=False, 
                            add_text='Residual', text_y_offset=0.7)
        for ax in [ax1, ax2, ax3]:
            ax.axis('off')
        plt.subplots_adjust(wspace=0.02)
        plt.savefig(output_name + '_result.png', bbox_inches='tight', facecolor='silver')
        plt.close()
        logger.info('Task finished! (⁎⁍̴̛ᴗ⁍̴̛⁎)')

        return results

    def _subtract_stackedPSF(self, results, res, halosize, bright_star_cat, median_psf, lowres_model, output_name):
        from astropy.coordinates import SkyCoord, match_coordinates_sky
        import astropy.units as u
        from mrf.celestial import Celestial, Star
        from mrf.utils import save_to_fits

        config = self.config
        logger = self.logger

        # 11. Build starhalo models and then subtract from "res" image
        logger.info('Draw star halo models onto the image, and subtract them!')
        # Make an extra edge, move stars right
        ny, nx = res.image.shape
        im_padded = np.zeros((ny + 2 * halosize, nx + 2 * halosize))
        # Making the left edge empty
        im_padded[halosize: ny + halosize, halosize: nx + halosize] = res.image
        im_halos_padded = np.zeros_like(im_padded)

        for i, obj in enumerate(bright_star_cat):
            spsf = Celestial(median_psf, header=lowres_model.header)
            x = obj['x']
            y = obj['y']
            x_int = x.astype(np.int)
            y_int = y.astype(np.int)
            dx = -1.0 * (x - x_int)
            dy = -1.0 * (y - y_int)
            spsf.shift_image(-dx, -dy, method=config.starhalo.interp)
            x_int, y_int = x_int + halosize, y_int + halosize
            if config.starhalo.norm == 'flux_ann':
                im_halos_padded[y_int - halosize:y_int + halosize + 1, 
                                x_int - halosize:x_int + halosize + 1] += spsf.image * obj['flux_ann']
            else:
                im_halos_padded[y_int - halosize:y_int + halosize + 1, 
                                x_int - halosize:x_int + halosize + 1] += spsf.image * obj['flux']

        im_halos = im_halos_padded[halosize: ny + halosize, halosize: nx + halosize]
        setattr(results, 'lowres_model_star', Celestial(im_halos, header=lowres_model.header))
        img_sub = res.image - im_halos
        setattr(results, 'lowres_final_unmask', Celestial(img_sub, header=res.header))
        lowres_model.image += im_halos
        setattr(results, 'lowres_model', lowres_model)

        save_to_fits(im_halos, '_lowres_halos.fits', header=lowres_model.header)
        save_to_fits(img_sub, output_name + '_halosub.fits', 
                        header=lowres_model.header)
        save_to_fits(lowres_model.image, output_name + '_model_halos.fits', 
                        header=lowres_model.header)
        logger.info('Bright star halos are subtracted!')
        return results

    def _subtract_widePSF(self, results, res, halosize, bright_star_cat, median_psf, lowres_model, output_name, skip_mast=False):
        from astropy.coordinates import SkyCoord, match_coordinates_sky
        import astropy.units as u
        from mrf.celestial import Celestial, Star
        from mrf.utils import save_to_fits

        config = self.config
        logger = self.logger

        #### 10.5: Build hybrid PSF with Qing's modelling.
        from .utils import save_to_fits
        from .utils import compute_Rnorm
        from .modeling import PSF_Model
        from photutils import CircularAperture
        
        ### PSF Parameters
        psf_size = 401                 # in pixel
        pixel_scale = config.lowres.pixel_scale  # in arcsec/pixel
        frac = 0.3                          # fraction of power law component (from fitting stacked PSF)
        beta = 3                            # moffat beta, in arcsec. This parameter is not used here. 
        fwhm = 2.28 * pixel_scale           # moffat fwhm, in arcsec. This parameter is not used here. 
        n0 = 3.24                           # first power-law index
        theta_0 = 5.                        # flattening radius (arbitrary), in arcsec. Inside which the power law is truncated.
        n_s = np.array([n0, 2.53, 1.22, 4])                          # power-law index
        theta_s = np.array([theta_0, 10**1.85, 10**2.18, 2 * psf_size])      # transition radius in arcsec
        ### Construct model PSF
        params = {"fwhm": fwhm, "beta": beta, "frac": frac, "n_s": n_s, 'theta_s': theta_s}
        psf = PSF_Model(params, aureole_model='multi-power')
        ### Build grid of image for drawing
        psf.make_grid(psf_size, pixel_scale)
        ### Generate the aureole of PSF
        psf_e, _ = psf.generate_aureole(psf_range=2 * psf_size)
        ### Hybrid radius (in pixel)
        try:
            hybrid_r = config.starhalo.hybrid_r
        except:
            hybrid_r = 12

        ### Inner PSF: from stacking stars
        inner_psf = copy.deepcopy(median_psf)
        inner_psf /= np.sum(inner_psf) # Normalize
        inner_size = inner_psf.shape   
        inner_cen = [int(x / 2) for x in inner_size]
        ##### flux_inn is the flux inside an annulus, we use this to scale inner and outer parts
        flux_inn = compute_Rnorm(inner_psf, None, inner_cen, R=hybrid_r, display=False, mask_cross=False)[1]
        ##### We only remain the stacked PSF inside hybrid radius. 
        aper = CircularAperture(inner_cen, hybrid_r).to_mask()
        mask = aper.to_image(inner_size) == 0
        inner_psf[mask] = np.nan

        ### Make new empty PSF
        outer_cen = (int(psf_size / 2), int(psf_size / 2))
        new_psf = np.zeros((int(psf_size), int(psf_size)))
        new_psf[outer_cen[0] - inner_cen[0]:outer_cen[0] + inner_cen[0] + 1, 
                outer_cen[1] - inner_cen[1]:outer_cen[1] + inner_cen[1] + 1] = inner_psf

        ### Outer PSF: from model
        outer_psf = psf_e.drawImage(nx=psf_size, ny=psf_size, scale=config.lowres.pixel_scale, method="no_pixel").array
        outer_psf /= np.sum(outer_psf) # Normalize
        ##### flux_out is the flux inside an annulus, we use this to scale inner and outer parts
        flux_out = compute_Rnorm(outer_psf, None, outer_cen, 
                                R=hybrid_r, display=False, mask_cross=False)[1]

        ##### Scale factor: the flux ratio near hybrid radius 
        scale_factor = flux_out / flux_inn
        temp = copy.deepcopy(outer_psf)
        new_psf[np.isnan(new_psf)] = temp[np.isnan(new_psf)] / scale_factor # fill `nan`s with the outer PSF
        temp[outer_cen[0] - inner_cen[0]:outer_cen[0] + inner_cen[0] + 1, 
                outer_cen[1] - inner_cen[1]:outer_cen[1] + inner_cen[1] + 1] = 0
        new_psf += temp / scale_factor
        new_psf /= np.sum(new_psf) # Normalize
        factor = np.sum(median_psf) / np.sum(new_psf[outer_cen[0] - inner_cen[0]:outer_cen[0] + inner_cen[0] + 1, 
                                                    outer_cen[1] - inner_cen[1]:outer_cen[1] + inner_cen[1] + 1])
        new_psf *= factor
        save_to_fits(new_psf, './wide_psf.fits')
        setattr(results, 'wide_PSF', new_psf)


        ### 11. Build starhalo models and then subtract from "res" image
        logger.info('Draw star halo models onto the image, and subtract them!')
        if skip_mast:
            ps1_cat = Table.read('./_ps1_cat.fits')
        else:
            ### Use Pan-STARRS catalog to normalize these bright stars
            from mrf.utils import ps1cone
            # Query PANSTARRS starts
            constraints = {'nDetections.gt':1, config.lowres.band + 'MeanPSFMag.lt':18}
            # strip blanks and weed out blank and commented-out values
            columns = """objID,raMean,decMean,raMeanErr,decMeanErr,nDetections,ng,nr,gMeanPSFMag,rMeanPSFMag""".split(',')
            columns = [x.strip() for x in columns]
            columns = [x for x in columns if x and not x.startswith('#')]
            logger.info('Retrieving Pan-STARRS catalog from MAST! Please wait!')
            ps1result = ps1cone(results.lowres_input.ra_cen, results.lowres_input.dec_cen, results.lowres_input.diag_radius.to(u.deg).value, 
                                release='dr2', columns=columns, verbose=False, **constraints)
            ps1_cat = Table.read(ps1result, format='csv')
            ps1_cat.add_columns([Column(data = lowres_model.wcs.wcs_world2pix(ps1_cat['raMean'], ps1_cat['decMean'], 0)[0], 
                                        name='x_ps1'),
                                Column(data = lowres_model.wcs.wcs_world2pix(ps1_cat['raMean'], ps1_cat['decMean'], 0)[1], 
                                        name='y_ps1')])
            ps1_cat = ps1_cat[ps1_cat[config.lowres.band + 'MeanPSFMag'] != -999]
            ps1_cat.write('./_ps1_cat.fits', overwrite=True)
            #ps1_cat = Table.read('./_ps1_cat.fits')

        ## Match PS1 catalog with SEP one
        temp, dist, _ = match_coordinates_sky(SkyCoord(ra=bright_star_cat['ra'], dec=bright_star_cat['dec'], unit='deg'),
                                            SkyCoord(ra=ps1_cat['raMean'], dec=ps1_cat['decMean'], unit='deg'))
        flag = dist < 5 * u.arcsec
        temp = temp[flag]
        reorder_cat = vstack([bright_star_cat[flag], bright_star_cat[~flag]], join_type='outer')
        bright_star_cat = hstack([reorder_cat, ps1_cat[temp]], join_type='outer')     
        bright_star_cat.write('_bright_star_cat.fits', format='fits', overwrite=True)
        setattr(results, 'bright_star_cat', bright_star_cat)

        ### Fit an empirical relation between PS1 magnitude and SEP flux
        from astropy.table import MaskedColumn
        if isinstance(bright_star_cat['rMeanPSFMag'], MaskedColumn):
            mask = (~bright_star_cat.mask[config.lowres.band + 'MeanPSFMag'])
            flag = (bright_star_cat[config.lowres.band + 'MeanPSFMag'] < 16) & mask
        else:
            flag = (bright_star_cat[config.lowres.band + 'MeanPSFMag'] < 16)
        x = bright_star_cat[flag][config.lowres.band + 'MeanPSFMag']
        y = -2.5 * np.log10(bright_star_cat[flag]['flux']) # or flux_ann
        pfit = np.polyfit(x, y, 2) # second-order polynomial
        plt.scatter(x, y, s=13)
        plt.plot(np.linspace(10, 16, 20), np.poly1d(pfit)(np.linspace(10, 16, 20)), color='red')
        plt.xlabel('MeanPSFMag')
        plt.ylabel('-2.5 Log(SE flux)')
        plt.savefig('./PS1-normalization.png')
        plt.close()

        # Make an extra edge, move stars right
        ny, nx = res.image.shape
        im_padded = np.zeros((int(ny + psf_size), int(nx + psf_size)))
        # Making the left edge empty
        im_padded[int(psf_size/2): ny + int(psf_size/2), int(psf_size/2): nx + int(psf_size/2)] = res.image
        im_halos_padded = np.zeros_like(im_padded)

        # Stack stars onto the canvas
        for i, obj in enumerate(bright_star_cat):
            spsf = Celestial(new_psf, header=lowres_model.header)
            x = obj['x']
            y = obj['y']
            x_int = x.astype(np.int)
            y_int = y.astype(np.int)
            dx = -1.0 * (x - x_int)
            dy = -1.0 * (y - y_int)
            spsf.shift_image(-dx, -dy, method=config.starhalo.interp)
            x_int, y_int = x_int + int(psf_size/2), y_int + int(psf_size/2)

            if obj['mag'] < 15.5:
                if obj[config.lowres.band + 'MeanPSFMag']:
                    norm = 10**((-np.poly1d(pfit)(obj[config.lowres.band + 'MeanPSFMag'])) / 2.5)
                else:
                    norm = obj['flux']
            else:
                norm = obj['flux']

            im_halos_padded[y_int - int(psf_size/2):y_int + int(psf_size/2) + 1, 
                            x_int - int(psf_size/2):x_int + int(psf_size/2) + 1] += spsf.image * norm

        im_halos = im_halos_padded[int(psf_size/2): ny + int(psf_size/2), int(psf_size/2): nx + int(psf_size/2)]

        model_star = Celestial(im_halos, header=lowres_model.header)

        model_star.shift_image(-1/config.lowres.magnify_factor, -1/config.lowres.magnify_factor, method=config.starhalo.interp)

        setattr(results, 'lowres_model_star', model_star)
        img_sub = res.image - model_star.image
        setattr(results, 'lowres_final_unmask', Celestial(img_sub, header=res.header))
        lowres_model.image += model_star.image
        setattr(results, 'lowres_model', lowres_model)

        save_to_fits(model_star.image, '_lowres_halos.fits', header=lowres_model.header)
        save_to_fits(img_sub, output_name + '_halosub.fits', 
                        header=lowres_model.header)
        save_to_fits(lowres_model.image, output_name + '_model_halos.fits', 
                        header=lowres_model.header)
        logger.info('Bright star halos are subtracted!')
        return results
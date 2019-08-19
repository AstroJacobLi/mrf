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
from astropy.table import Table, Column, hstack

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

class Results():
    """
    Results class.
    """
    def __init__(self, config):
        self.config = config
    
class MrfTask():
    '''
    MRF task class. This class implements `mrf`.
    '''
    def __init__(self, config_file):
        # Open configuration file
        with open(config_file, 'r') as ymlfile:
            cfg = yaml.safe_load(ymlfile)
            config = Config(cfg)
        self.config_file = config_file
        self.config = config

    def set_logger(self, verbose=True):
        
        if verbose:
            log_filename = self.config_file.rstrip('yaml') + 'log'
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
            output_name='mrf', verbose=True):
        """
        Run MRF task.

        Parameters:
            dir_lowres: string, directory of input low-resolution image.
            dir_hires_b: string, directory of input high-resolution blue-band image (typically g-band).
            dir_hires_r: string, directory of input high-resolution red-band image (typically r-band).
            certain_gal_cat: string, directory of a catalog (in ascii format) which contains 
                RA and DEC of galaxies which you want to retain during MRF.
            output_name: string, which will be the prefix of output files.
            verbose: bool. If True, it will make a log file recording the process. 

        Returns:
            results: `Results` class, containing key results of this task.
        
        """
        from astropy.coordinates import SkyCoord, match_coordinates_sky
        from astropy.convolution import convolve, Box2DKernel, Gaussian2DKernel
        from mrf.utils import (save_to_fits, Flux_Model, mask_out_stars, extract_obj, \
                            bright_star_mask, Autokernel, psf_bkgsub)
        from mrf.utils import seg_remove_obj, mask_out_certain_galaxy

        from mrf.display import display_single, SEG_CMAP, display_multiple, draw_circles
        from mrf.celestial import Celestial, Star
        from mrf.utils import Config
        from reproject import reproject_interp

        config = self.config
        logger = self.set_logger(verbose=verbose)
        results = Results(config)

        logger.info('Running Multi-Resolution Filtering (MRF) on "{0}" and "{1}" images!'.format(config.hires.dataset, config.lowres.dataset))
        setattr(results, 'lowres_name', config.lowres.dataset)
        setattr(results, 'hires_name', config.hires.dataset)

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
        lowres.resize_image(f_magnify)
        lowres.save_to_fits('_lowres_{}.fits'.format(int(f_magnify)))
        logger.info('Register high resolution image "{0}" with "{1}"'.format(dir_hires_b, dir_lowres))
        hdu = fits.open(dir_hires_b)
        if 'hsc' in dir_hires_b:
            array, _ = reproject_interp(hdu[1], lowres.header)
        else:
            array, _ = reproject_interp(hdu[0], lowres.header)
        hires_b = Celestial(array, header=lowres.header)
        hdu.close()
        
        logger.info('Register high resolution image "{0}" with "{1}"'.format(dir_hires_r, dir_lowres))
        hdu = fits.open(dir_hires_r)
        if 'hsc' in dir_hires_r:
            array, _ = reproject_interp(hdu[1], lowres.header)
        else:
            array, _ = reproject_interp(hdu[0], lowres.header)
        hires_r = Celestial(array, header=lowres.header)
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
        
        logger.info('Build flux models on high-resolution images: Blue band')
        logger.info('    - sigma = %.1f, minarea = %d', sigma, minarea)
        logger.info('    - deblend_cont = %.5f, deblend_nthres = %.1f', deblend_cont, deblend_nthresh)
        _, _, b_imflux = Flux_Model(hires_b.image, hires_b.header, sigma=sigma, minarea=minarea, 
                                    deblend_cont=deblend_cont, deblend_nthresh=deblend_nthresh, 
                                    save=True, logger=logger)
        
        logger.info('Build flux models on high-resolution images: Red band')
        logger.info('    - sigma = %.1f, minarea = %d', sigma, minarea)
        logger.info('    - deblend_cont = %.5f, deblend_nthres = %.1f', deblend_cont, deblend_nthresh)
        _, _, r_imflux = Flux_Model(hires_r.image, hires_b.header, sigma=sigma, minarea=minarea, 
                                    deblend_cont=deblend_cont, deblend_nthresh=deblend_nthresh, 
                                    save=True, logger=logger)
        

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
        
        # Clear memory
        del r_imflux, b_imflux, hires_b, hires_r
        gc.collect()
        
        # 5. Extract sources on hi-res corrected image
        logger.info('Extract objects from color-corrected high resolution image with:')
        logger.info('    - sigma = %.1f, minarea = %d', sigma, minarea)
        logger.info('    - deblend_cont = %.5f, deblend_nthres = %.1f', deblend_cont, deblend_nthresh)
        objects, segmap = extract_obj(hires_3.image, b=b, f=f, sigma=sigma, minarea=minarea, 
                                      show_fig=False, flux_aper=flux_aper, 
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
                                        show_figure=config.kernel.show_fig, cval=0.0,
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
                            fill_value=0, nan_treatment='fill', normalize_kernel=False)
        save_to_fits(conv_model, '_lowres_model_{}.fits'.format(int(f_magnify)), header=hires_3.header)
        
        # Optinally remove low surface brightness objects from model: 
        if config.fluxmodel.unmask_lowsb:
            logger.info('    - Removing low-SB objects from flux model.')
            from .utils import remove_lowsb
            hires_flxmd = remove_lowsb(hires_fluxmod.image, conv_model, kernel_med, seg, 
                                        "_hires_obj_cat.fits", 
                                        SB_lim=config.fluxmodel.sb_lim, 
                                        zeropoint=config.hires.zeropoint, 
                                        pixel_size=config.hires.pixel_scale / f_magnify, 
                                        unmask_ratio=config.fluxmodel.unmask_ratio, 
                                        gaussian_radius=config.fluxmodel.gaussian_radius, 
                                        gaussian_threshold=config.fluxmodel.gaussian_threshold, 
                                        logger=logger)

            logger.info('    - Convolving image, this could be a bit slow @_@')
            conv_model = convolve_fft(hires_flxmd, kernel_med, boundary='fill', 
                                 fill_value=0, nan_treatment='fill', normalize_kernel=False)
            save_to_fits(conv_model, '_lowres_model_clean_{}.fits'.format(f_magnify), header=hires_3.header)
            setattr(results, 'hires_fluxmod', hires_flxmd)

        lowres_model = Celestial(conv_model, header=hires_3.header)
        res = Celestial(lowres.image - lowres_model.image, header=lowres.header)
        res.save_to_fits('_res_{}.fits'.format(f_magnify))

        lowres_model.resize_image(1 / f_magnify)
        lowres_model.save_to_fits('_lowres_model.fits')
        setattr(results, 'lowres_model_compact', copy.deepcopy(lowres_model))

        res.resize_image(1 / f_magnify)
        res.save_to_fits(output_name + '_res.fits')
        setattr(results, 'res', res)

        logger.info('Compact objects has been subtracted from low-resolution image! Saved as "{}".'.format(output_name + '_res.fits'))


        #10. Subtract bright star halos! Only for those left out in flux model!
        star_cat = Table.read('_bright_stars_3.fits', format='fits')
        star_cat['x'] /= f_magnify
        star_cat['y'] /= f_magnify
        ra, dec = res.wcs.wcs_pix2world(star_cat['x'], star_cat['y'], 0)
        star_cat.add_columns([Column(data=ra, name='ra'), Column(data=dec, name='dec')])

        sigma = config.starhalo.sigma
        minarea = config.starhalo.minarea
        deblend_cont = config.starhalo.deblend_cont
        deblend_nthresh = config.starhalo.deblend_nthresh
        sky_subtract = config.starhalo.sky_subtract
        flux_aper = config.starhalo.flux_aper

        logger.info('Extract objects from compact-object-subtracted low-resolution image with:')
        logger.info('    - sigma = %.1f, minarea = %d', sigma, minarea)
        logger.info('    - deblend_cont = %.5f, deblend_nthres = %.1f', deblend_cont, deblend_nthresh)
        objects, segmap = extract_obj(res.image.byteswap().newbyteorder(), 
                                    b=64, f=3, sigma=sigma, minarea=minarea,
                                    deblend_nthresh=deblend_nthresh, 
                                    deblend_cont=deblend_cont, 
                                    sky_subtract=sky_subtract, show_fig=False, 
                                    flux_aper=flux_aper, logger=logger)
        
        ra, dec = res.wcs.wcs_pix2world(objects['x'], objects['y'], 0)
        objects.add_columns([Column(data=ra, name='ra'), Column(data=dec, name='dec')])
        
        # Match two catalogs
        logger.info('Stack stars to get PSF model!')
        logger.info('    - Match detected objects with previously discard stars')
        temp = match_coordinates_sky(SkyCoord(ra=star_cat['ra'], dec=star_cat['dec'], unit='deg'),
                                     SkyCoord(ra=objects['ra'], dec=objects['dec'], unit='deg'))[0]
        bright_star_cat = objects[np.unique(temp)]
        mag = config.lowres.zeropoint - 2.5 * np.log10(bright_star_cat['flux'])
        bright_star_cat.add_column(Column(data=mag, name='mag'))
        bright_star_cat.write('_bright_star_cat.fits', format='fits', overwrite=True)

        # Select good stars to stack
        psf_cat = bright_star_cat[bright_star_cat['fwhm_custom'] < config.starhalo.fwhm_lim] # FWHM selection
        psf_cat = psf_cat[psf_cat['mag'] < config.starhalo.bright_lim]
        psf_cat.sort('flux')
        psf_cat.reverse()
        psf_cat = psf_cat[:int(config.starhalo.n_stack)]
        logger.info('    - Get {} stars to be stacked!'.format(len(psf_cat)))

        # Construct and stack `Stars`.
        halosize = config.starhalo.halosize
        padsize = config.starhalo.padsize
        size = 2 * halosize + 1
        stack_set = np.zeros((len(psf_cat), size, size))
        bad_indices = []
        for i, obj in enumerate(psf_cat):
            try:
                sstar = Star(res.image, header=res.header, starobj=obj, 
                             halosize=halosize, padsize=padsize)
                if config.starhalo.mask_contam:
                    sstar.mask_out_contam(show_fig=False, verbose=False)
                sstar.centralize(method='iraf')
                #sstar.sub_bkg(verbose=False)
                cval = config.starhalo.cval
                if isinstance(cval, str) and 'nan' in cval.lower():
                    cval = np.nan
                else:
                    cval = float(cval)

                if config.starhalo.norm == 'flux_ann':
                    stack_set[i, :, :] = sstar.get_masked_image(cval=cval) / sstar.fluxann
                else:
                    stack_set[i, :, :] = sstar.get_masked_image(cval=cval) / sstar.flux
                    
            except Exception as e:
                stack_set[i, :, :] = np.ones((size, size)) * 1e9
                bad_indices.append(i)
                logger.info(e)
                
        stack_set = np.delete(stack_set, bad_indices, axis=0)
        median_psf = np.nanmedian(stack_set, axis=0)
        median_psf = psf_bkgsub(median_psf, int(config.starhalo.edgesize))
        median_psf = convolve(median_psf, Box2DKernel(3))
        save_to_fits(median_psf, '_median_psf.fits');
        setattr(results, 'PSF', median_psf)

        logger.info('    - Stars are stacked successfully!')
        save_to_fits(stack_set, '_stack_bright_stars.fits')

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
            spsf.shift_image(-dx, -dy, method='iraf')
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
        lowres_model.image += im_halos
        setattr(results, 'lowres_model', lowres_model)

        save_to_fits(im_halos, '_lowres_halos.fits', header=lowres_model.header)
        save_to_fits(img_sub, output_name + '_halosub.fits', 
                     header=lowres_model.header)
        save_to_fits(lowres_model.image, output_name + '_model_halos.fits', 
                     header=lowres_model.header)
        
        logger.info('Bright star halos are subtracted!')


        # 11. Mask out dirty things!
        if config.clean.clean_img:
            logger.info('Clean the image!')
            model_mask = convolve(lowres_model.image, 
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
                from compsub.utils import img_replace_with_noise
                final_image = img_replace_with_noise(img_sub.byteswap().newbyteorder(), totmask)
            else:
                logger.info('    - Replace artifacts with void.')
                final_image = img_sub * (~totmask.astype(bool))
            
            save_to_fits(final_image, output_name + '_final.fits', header=res.header)
            setattr(results, 'lowres_final', Celestial(final_image, header=res.header))
            logger.info('The final result is saved as "{}"!'.format(output_name + '_final.fits'))
        # Delete temp files
        if config.clean.clean_file:
            logger.info('Delete all temporary files!')
            os.system('rm -rf _*.fits')


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
        logger.info('Mission finished! (⁎⁍̴̛ᴗ⁍̴̛⁎)')

        return results
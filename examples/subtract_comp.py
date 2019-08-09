import sys
import os
import math
import logging
import copy
import yaml
import argparse
import logging
import numpy as np
import matplotlib.pyplot as plt

from astropy import wcs
from astropy.io import fits
from astropy.coordinates import SkyCoord, match_coordinates_sky
from astropy.table import Table, Column, hstack
from mrf.utils import (save_to_fits, Flux_Model, mask_out_stars, extract_obj, \
                        star_mask, Autokernel, query_star, psf_bkgsub)
from mrf.utils import seg_remove_obj, mask_out_certain_galaxy

from mrf.display import display_single, SEG_CMAP, display_multiple, draw_circles
from mrf.celestial import Celestial #, Star
from reproject import reproject_interp

class Config(object):
    def __init__(self, d):
        for a, b in d.items():
            if isinstance(b, (list, tuple)):
                setattr(self, a, [Config(x) if isinstance(x, dict) else x for x in b])
            else:
                setattr(self, a, Config(b) if isinstance(b, dict) else b)

def main(argv=sys.argv[1:]):
    # In non-script code, use getLogger(__name__) at module scope instead.
    logging.basicConfig(format='%(asctime)s %(message)s', level=logging.INFO, 
                        handlers=[logging.StreamHandler(sys.stdout),
                                  logging.FileHandler('DF_compsub.log', mode='w')])
    logger = logging.getLogger("DF_compsub.log")
    # Parse the input
    parser = argparse.ArgumentParser(description='This script subtract compact objects from Dragonfly image.')
    parser.add_argument('--config', '-c', required=True, help='configuration file')
    args = parser.parse_args()
    #############################################################
    #############################################################
    logger.info('Open configuration file {}'.format(args.config))
    # Open configuration file
    with open(args.config, 'r') as ymlfile:
        cfg = yaml.safe_load(ymlfile)
        config = Config(cfg)

    # 1. subtract background of DF, if desired
    df_image = config.file.df_image
    hi_res_image_blue = config.file.hi_res_image_blue
    hi_res_image_red = config.file.hi_res_image_red
    
    hdu = fits.open(config.file.df_image)
    df = Celestial(hdu[0].data, header=hdu[0].header)
    if config.DF.sub_bkgval:
        logger.info('Subtract BACKVAL=%.1f of Dragonfly image', float(df.header['BACKVAL']))
        df.image -= float(df.header['BACKVAL'])
    hdu.close()

    # 2. Create magnified DF image, and register high-res images with subsampled DF ones
    f_magnify = config.DF.magnify_factor
    resize_method = config.DF.resize_method
    logger.info('Magnify Dragonfly image with a factor of %.1f:', f_magnify)
    df.resize_image(f_magnify, method=resize_method);
    df.save_to_fits('_df_{}.fits'.format(int(f_magnify)));
    logger.info('Register high resolution image {} with Dragonfly image'.format(hi_res_image_blue))
    hdu = fits.open(hi_res_image_blue)
    if 'hsc' in hi_res_image_blue:
        array, _ = reproject_interp(hdu[1], df.header)
    else:
        array, _ = reproject_interp(hdu[0], df.header)
    hires_b = Celestial(array, header=df.header)
    hdu.close()
    logger.info('Register high resolution image {} with Dragonfly image'.format(hi_res_image_red))
    hdu = fits.open(hi_res_image_red)
    if 'hsc' in hi_res_image_red:
        array, _ = reproject_interp(hdu[1], df.header)
    else:
        array, _ = reproject_interp(hdu[0], df.header)
    hires_r = Celestial(array, header=df.header)
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
                                deblend_cont=deblend_cont, deblend_nthresh=deblend_nthresh, save=True)
    
    logger.info('Build flux models on high-resolution images: Red band')
    logger.info('    - sigma = %.1f, minarea = %d', sigma, minarea)
    logger.info('    - deblend_cont = %.5f, deblend_nthres = %.1f', deblend_cont, deblend_nthresh)
    _, _, r_imflux = Flux_Model(hires_r.image, hires_b.header, sigma=sigma, minarea=minarea, 
                                deblend_cont=deblend_cont, deblend_nthresh=deblend_nthresh, save=True)
    

    # 4. Make color correction to blue band, remove artifacts as well
    logger.info('Make color correction to blue band, remove artifacts as well')
    col_ratio = (b_imflux / r_imflux)
    col_ratio[np.isnan(col_ratio) | np.isinf(col_ratio)] = 0 # remove artifacts
    save_to_fits(col_ratio, '_colratio.fits', header=hires_b.header)
    
    color_term = config.DF.color_term
    logger.info('### color_term = ' + str(color_term))
    median_col = np.nanmedian(col_ratio[col_ratio != 0])
    logger.info('### median_color (b/r) = ' + str(round(median_col, 5)))

    fluxratio = col_ratio / median_col
    fluxratio[(fluxratio < 0.1) | (fluxratio > 10)] = 1 # does this make sense?
    col_correct = np.power(fluxratio, color_term) # how to improve this correction?
    save_to_fits(col_correct, '_colcorrect.fits', header=hires_b.header)

    if config.DF.band == 'r':
        hires_3 = Celestial(hires_r.image, header=hires_b.header)
    elif config.DF.band == 'g':
        hires_3 = Celestial(hires_b.image * col_correct, header=hires_b.header)
    else:
        raise ValueError('config.DF.band must be "g" or "r"!')

    _ = hires_3.save_to_fits('_hires_{}.fits'.format(int(f_magnify)))
    

    # 5. Extract sources on hires corrected image
    logger.info('Extracting objects from color-corrected high resolution image with:')
    logger.info('    - sigma = %.1f, minarea = %d', sigma, minarea)
    logger.info('    - deblend_cont = %.5f, deblend_nthres = %.1f', deblend_cont, deblend_nthresh)
    objects, segmap = extract_obj(hires_3.image, b=b, f=f, sigma=sigma, minarea=minarea, 
                                  show_fig=False, flux_aper=flux_aper, 
                                  deblend_nthresh=deblend_nthresh, 
                                  deblend_cont=deblend_cont)
    objects.write('_hires_obj_cat.fits', format='fits', overwrite=True)
    
    # 6. Remove bright stars (and certain galaxies)
    logger.info('Remove bright stars from this segmentation map, using SEP results. ')
    logger.info('Bright star limit = {}'.format(config.star.bright_lim))
    seg = copy.deepcopy(segmap)
    mag = config.file.hi_res_zp - 2.5 * np.log10(abs(objects['flux']))
    flag = np.where(mag < config.star.bright_lim)
    for obj in objects[flag]:
        seg = seg_remove_obj(seg, obj['x'], obj['y'])

    #seg_gaia = mask_out_stars(segmap, hires_3.image, hires_3.header, 
    #                          method=config.star.method, bright_lim=config.star.bright_lim)
    
    # You can Mask out certain galaxy here.
    logger.info('Remove objects from catalog {}'.format(config.file.certain_gal_cat))
    gal_cat = Table.read(config.file.certain_gal_cat, format='ascii')
    seg = mask_out_certain_galaxy(seg, hires_3.header, gal_cat=gal_cat)
    
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
    _ = hires_fluxmod.save_to_fits('_hires_fluxmod.fits')
    logger.info('Flux model from high resolution image has been built!')
    
    # 8. Build kernel based on some stars
    img_hires = Celestial(hires_3.image.byteswap().newbyteorder(), 
                      header=hires_3.header, dataset='cfht_3')
    img_lowres = Celestial(df.image.byteswap().newbyteorder(), 
                       header=df.header, dataset='df_3')
    cval = config.kernel.cval

    if isinstance(cval, str) and 'nan' in cval.lower():
        cval = np.nan
    else:
        cval = float(cval)
        
    logger.info('Build convolving kernel to degrade high resolution image.')
    kernel_med, good_cat = Autokernel(img_hires, img_lowres, 
                                      int(f_magnify * config.kernel.kernel_size), 
                                      int(f_magnify * (config.kernel.kernel_size - config.kernel.kernel_edge)), 
                                      frac_maxflux=config.kernel.frac_maxflux, 
                                      show_figure=config.kernel.show_fig, cval=cval,
                                      nkernels=config.kernel.nkernel)
    # You can also circularize the kernel
    if config.kernel.circularize:
        logger.info('Circularize the kernel.')
        from compsub.utils import circularize
        kernel_med = circularize(kernel_med, n=14)
    save_to_fits(kernel_med, 'kernel_median.fits')
    
    
    # 9. Convolve this kernel to high-res image
    # Two options: if you have `galsim` installed, use galsim, it's much faster. 
    # Otherwise, use `fconvolve` from iraf.
    # Galsim solution:
    import galsim
    psf = galsim.InterpolatedImage(galsim.Image(kernel_med), 
                                   scale=config.DF.pixel_scale / f_magnify)
    gal = galsim.InterpolatedImage(galsim.Image(hires_fluxmod.image), 
                                   scale=config.DF.pixel_scale / f_magnify)
    logger.info('Convolving image, this will be a bit slow @_@ ###')
    final = galsim.Convolve([gal, psf])
    image = final.drawImage(scale=config.DF.pixel_scale / f_magnify, 
                            nx=hires_3.shape[1], 
                            ny=hires_3.shape[0])
    save_to_fits(image.array, '_df_model_{}.fits'.format(int(f_magnify)), header=hires_3.header)
    df_model = image.array

    # Optinally remove low surface brightness objects from model: 
    if config.fluxmodel.unmask_lowsb:
        E = hires_fluxmod.image / df_model
        E[np.isinf(df_model)] = 0.0
        E[np.isinf(E) | np.isnan(E)] = 0.0

        kernel_flux = np.sum(kernel_med)
        print("# Kernel flux = {}".format(kernel_flux))
        E *= kernel_flux
        print('# Maximum of E = {}'.format(np.nanmax(E)))

        im_seg = copy.deepcopy(seg)
        im_highres = copy.deepcopy(hires_fluxmod.image)
        im_ratio = E
        im_highres_new =  np.zeros_like(hires_fluxmod.image)
        objects = Table.read('_hires_obj_cat.fits', format='fits')
        
        # calculate SB limit in counts per pixel
        sb_lim_cpp = 10**((config.fluxmodel.sb_lim - config.DF.zeropoint)/(-2.5)) * (config.DF.pixel_scale * f_magnify)**2
        print('# SB limit in counts / pixel = {}'.format(sb_lim_cpp))
        
        im_seg_ind = np.where(im_seg>0)
        im_seg_slice = im_seg[im_seg_ind]
        im_highres_slice = im_highres[im_seg_ind]
        im_highres_new_slice = im_highres_new[im_seg_ind]
        im_ratio_slice = im_ratio[im_seg_ind]

        # loop over objects
        for obj in objects:
            ind = np.where(np.isin(im_seg_slice, obj['index']))
            flux_hires = im_highres_slice[ind]
            flux_ratio = im_ratio_slice[ind]
            if ((np.mean(flux_hires) < sb_lim_cpp) and (np.mean(flux_ratio) < config.fluxmodel.unmask_ratio)) and (np.mean(flux_ratio) != 0):
                im_highres_new_slice[ind] = 1
                print('# removed object {}'.format(obj['index']))
        
        im_highres_new[im_seg_ind] = im_highres_new_slice
        save_to_fits(im_highres_new, '_hires_fluxmode_clean_mask.fits')

        # BLow up
        # Then blow mask up
        smooth_radius = config.fluxmodel.gaussian_radius
        mask_conv = copy.deepcopy(im_highres_new)
        mask_conv[mask_conv > 0] = 1
        mask_conv = convolve(mask_conv.astype(float), Gaussian2DKernel(smooth_radius))
        seg_mask = (mask_conv >= config.fluxmodel.gaussian_threshold)
        im_highres[seg_mask] = 0

        psf = galsim.InterpolatedImage(galsim.Image(kernel_med), 
                                    scale=config.DF.pixel_scale / f_magnify)
        gal = galsim.InterpolatedImage(galsim.Image(im_highres), 
                                    scale=config.DF.pixel_scale / f_magnify)
        logger.info('Convolving image, this will be a bit slow @_@ ###')
        final = galsim.Convolve([gal, psf])
        image = final.drawImage(scale=config.DF.pixel_scale / f_magnify, 
                                nx=hires_3.shape[1], 
                                ny=hires_3.shape[0])
        save_to_fits(image.array, '_df_model_clean_{}.fits'.format(f_magnify), header=hires_3.header)

    df_model = Celestial(image.array, header=hires_3.header)
    res = Celestial(df.image - df_model.image, header=df.header)
    res.save_to_fits('_res_{}.fits'.format(f_magnify))
    
    df_model.resize_image(1 / f_magnify, method=resize_method)
    df.save_to_fits('_df_model.fits')
    
    res.resize_image(1 / f_magnify, method=resize_method)
    res.save_to_fits('res.fits')
    logger.info('Compact objects has been subtracted from Dragonfly image! Saved as "res.fits".')
    
    return 

    #### Subtract bright star halos!
    star_cat = query_star(res.image, res.header, method=config.starhalo.method, 
                          bright_lim=config.star.bright_lim + 0.5)
    x, y = res.wcs.wcs_world2pix(star_cat['ra'], star_cat['dec'], 0)
    fov_mask = (x < res.shape[1]) & (x > 0) & (y < res.shape[0]) & (y > 0)
    star_cat = star_cat[fov_mask]
    # Extract stars from image
    sigma = config.starhalo.sigma
    minarea = config.starhalo.minarea
    deblend_cont = config.starhalo.deblend_cont
    deblend_nthresh = config.starhalo.deblend_nthresh
    sky_subtract = config.starhalo.sky_subtract
    flux_aper = config.starhalo.flux_aper
    show_fig = config.starhalo.show_fig
    logger.info('Extracting objects from compact-object-corrected Dragonfly image with:')
    logger.info('    - sigma = %.1f, minarea = %d', sigma, minarea)
    logger.info('    - deblend_cont = %.5f, deblend_nthres = %.1f', deblend_cont, deblend_nthresh)
    objects, segmap = extract_obj(res.image.byteswap().newbyteorder(), 
                                  b=64, f=3, sigma=sigma, minarea=minarea,
                                  deblend_nthresh=deblend_nthresh, 
                                  deblend_cont=deblend_cont, 
                                  sky_subtract=sky_subtract, show_fig=show_fig, 
                                  flux_aper=flux_aper)
    ra, dec = res.wcs.wcs_pix2world(objects['x'], objects['y'], 0)
    objects.add_columns([Column(data=ra, name='ra'), Column(data=dec, name='dec')])
    # Match two catalogs
    logger.info('Match detected objects with {} catalog to ensure they are stars.'.format(config.starhalo.method))
    temp = match_coordinates_sky(SkyCoord(ra=star_cat['ra'], dec=star_cat['dec'], unit='deg'),
                                 SkyCoord(ra=objects['ra'], dec=objects['dec'], unit='deg'))[0]
    psf_cat = hstack([objects[temp], star_cat], join_type='exact') # here's the final star catalog
    psf_cat = psf_cat[psf_cat['fwhm_custom'] < config.starhalo.fwhm_lim] # FWHM selection
    psf_cat = psf_cat[psf_cat['phot_g_mean_mag'] < config.starhalo.bright_lim]
    psf_cat.sort('flux')
    psf_cat.reverse()
    psf_cat = psf_cat[:int(config.starhalo.n_stack)]
    logger.info('You get {} stars to be stacked!'.format(len(psf_cat)))
    # Construct and stack `Stars`!!!.
    halosize = config.starhalo.halosize
    padsize = config.starhalo.padsize
    size = 2 * halosize + 1
    stack_set = np.zeros((len(psf_cat), size, size))
    bad_indices = []
    logger.info('Stacking stars!')
    for i, obj in enumerate(psf_cat):
        try:
            sstar = StackStar(res.image, header=res.header, starobj=obj, 
                              halosize=halosize, padsize=padsize)
            sstar.mask_out_contam(method='sep', show_fig=False, verbose=False)
            sstar.centralize(order=5)
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
    save_to_fits(median_psf, 'median_psf.fits');
    logger.info('Stars are stacked in to a PSF and saved as "median_psf.fits"!')
    
    ## Build starhalo models and then subtract
    logger.info('Draw star halo models onto the image, and subtract them!')
    zp = df.header['MEDIANZP'] 
    objects = objects[objects['flux'] > 0]
    objects.sort('flux')
    objects.reverse()

    flux = objects['flux'].data
    flux_annulus = objects['flux_ann'].data
    fwhm = objects['fwhm_custom'].data
    mag = zp - 2.5 * np.log10(np.abs(flux))
    sel = np.where((mag < 19.5) & (fwhm < 25))[0]
    # Make an extra edge, move stars right
    ny, nx = res.image.shape
    im_padded = np.zeros((ny + 2 * halosize, nx + 2 * halosize))
    # Making the left edge empty
    im_padded[halosize: ny + halosize, halosize: nx + halosize] = res.image
    im_halos_padded = np.zeros_like(im_padded)

    for i, obj in enumerate(objects[sel]):
        spsf = Stack(median_psf, header=df.header)
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
    img_sub = res.image - im_halos
    save_to_fits(img_sub, 'df_halosub.fits')
    logger.info('Bright star halos are subtracted! Saved as "df_halosub.fits".')

    # Mask out dirty things!
    if config.clean.clean_img:
        logger.info('Clean the image! Replace relics with noise.')
        model_mask = convolve(df_model.image, Gaussian2DKernel(config.clean.gaussian_radius))
        model_mask[model_mask < 10] = 0
        model_mask[model_mask != 0] = 1
        strmask = star_mask(segmap, res.image, res.header, 
                            method=config.clean.star_method, bright_lim=config.clean.bright_lim)
        #strmask = convolve(strmask, Box2DKernel(10))
        # Total mask with noise
        totmask = model_mask + strmask
        totmask[totmask > 0] = 1
        totmask = convolve(totmask, Box2DKernel(2))
        totmask[totmask > 0] = 1
        if config.clean.replace_with_noise:
            from compsub.utils import img_replace_with_noise
            final_image = img_replace_with_noise(res.image.byteswap().newbyteorder(), totmask)
        else:
            final_image = res.image * (~totmask.astype(bool))
        save_to_fits(final_image, 'final_image.fits', header=res.header)
        logger.info('The final result is saved as "final_image.fits"!')
    # Delete temp files
    if config.clean.clean_file:
        logger.info('Delete all temporary files!')
        os.system('rm -rf _*.fits')

    logger.info('Mission finished!')

if __name__ == "__main__":
    main()
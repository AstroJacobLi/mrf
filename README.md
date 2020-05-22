# MRF: Multi-Resolution Filtering
Multi-Resolution Filtering: a method for isolating faint, extended emission in [Dragonfly](http://dragonflytelescope.org) data and other low resolution images.

![](https://readthedocs.org/projects/mrfiltering/badge/?version=latest)
![](https://img.shields.io/badge/license-MIT-blue)
![](https://img.shields.io/github/repo-size/astrojacobli/mrf)
[![](https://img.shields.io/badge/arXiv-1910.12867-blue)](https://arxiv.org/abs/1910.12867)

<p align="center">
  <img src="https://github.com/AstroJacobLi/mrf/blob/master/df-logo.png" width="40%">
</p>

Documentation
-------------
Please read the documentation and tutorial at https://mrfiltering.readthedocs.io/en/latest/.

Applications
------------
- Subtract compact objects from low-resolution images (such as Dragonfly) to reveal low surface brightness features.
- Characterize and subtract stellar halos in Dragonfly image.
- Download corresponding high resolution image (HSC, CFHT) of given Dragonfly image.

Examples
------------
This example shows the tidal feature of NGC 5907, described in [van Dokkum et al. (2019)](https://ui.adsabs.harvard.edu/abs/2019arXiv190611260V/abstract). The images presented there used this algorithm. Full resolution Dragonfly images and MRF results can be found [here](https://www.pietervandokkum.com/ngc5907). Check [this notebook](https://github.com/AstroJacobLi/mrf/blob/master/examples/NGC5907/mrfTask-n5907.ipynb) for more details in how to do MRF using this Python package! :rocket: 

![MRF on NGC 5907](https://github.com/AstroJacobLi/mrf/raw/master/examples/NGC5907/n5907-demo.png)

This example shows how powerful MRF is in revealing low surface brightness features. The ultra-diffuse galaxy M101-DF3 is revealed by MRF after subtracting compact objects and bright star halos according to [van Dokkum et al. (2019)](https://arxiv.org/abs/1910.12867). Check [this notebook](https://github.com/AstroJacobLi/mrf/blob/master/examples/M101-DF3/mrfTask-m101df3.ipynb) for more details.

![MRF on M101-DF3](https://github.com/AstroJacobLi/mrf/raw/master/examples/M101-DF3/m101-df3-demo.png)

You can also use [this script](https://github.com/AstroJacobLi/mrf/blob/master/examples/mrf-task.py) to run the MRF task. Take NGC 5907 as an example:

```bash
python mrf-task.py n5907_df_g.fits ngc5907_cfht_g.fits ngc5907_cfht_r.fits ngc5907-task.yaml --galcat='gal_cat_n5907.txt' --output='n5907_g'
```

Installation
------------

```bash
mkdir <install dir>
cd <install dir>
git clone git@github.com:AstroJacobLi/mrf.git
cd mrf
python setup.py install
```

If you don't have `git` configured, you can also download the `zip` file directly from https://github.com/AstroJacobLi/mrf/archive/master.zip, then unzip it and install in the same way. 

To test whether `mrf` is installed successfully, import `mrf` in Python:

```python
import mrf, os
print(os.path.isfile(os.path.join(mrf.__path__[0], 'iraf/macosx/x_images.e')))
```
`True` means you have installed `mrf` successfully! Bravo!

You must have [``galsim``](https://github.com/GalSim-developers/GalSim) installed in advance. You will also need [``unagi``](https://github.com/dr-guangtou/unagi) to download HSC images. `Python>=3` is needed, but you can try whether `mrf` still works under `python2`. Check out the dependence of `mrf` depends from `requirements.txt`.

Citation
---------
``mrf`` is a free software made available under the MIT License by [Pieter van Dokkum](http://pietervandokkum.com) (initial development) and [Jiaxuan Li](https://astrojacobli.github.io) (implementation, maintenance, and documentation). If you use this package in your work, please cite [van Dokkum et al. (2019)](https://arxiv.org/abs/1910.12867).

Acknowledgement
---------------
Many scripts and snippets are from [`kungpao`](https://github.com/dr-guangtou/kungpao) (written by [Song Huang](http://dr-guangtou.github.io)). [Qing Liu (U Toronto)](http://astro.utoronto.ca/~qliu) provided very useful functions for constructing wide-angle PSF, [Johnny Greco](http://johnnygreco.github.io) kindly shared his idea of the code structure. [Roberto Abraham](http://www.astro.utoronto.ca/~abraham/Web/Welcome.html) found the first few bugs of this package and provided useful solutions. Here we appreciate their help!

This package makes use of [`numpy`](http://www.numpy.org), [`scipy`](https://www.scipy.org), [`matplotlib`](https://matplotlib.org), [`Astropy`](http://www.astropy.org), [`sep`](https://sep.readthedocs.io/en/v1.0.x/), [`SWarp`](http://www.astromatic.net/software/swarp), [`GalSim`](http://galsim-developers.github.io/GalSim/_build/html/index.html), [`Photutils`](https://photutils.readthedocs.io/en/stable/), [`reproject`](https://reproject.readthedocs.io), [`IRAF (STScI)`](https://iraf-community.github.io). We thank the authors of these tools for their great efforts.


Copyright 2019 [Pieter van Dokkum](http://pietervandokkum.com) and [Jiaxuan Li](http://astrojacobli.github.io).
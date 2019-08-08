# MRF: Multi-Resolution Filtering
Multi-Resolution Filtering: a method for isolating faint, extended emission in [Dragonfly](http://dragonflytelescope.org) data and other low resolution images.

<p align="center">
  <img src="df-logo.png" width="40%">
</p>

------------

Applications
------------
- Download cooresponding high resolution image (HSC, CFHT) of given Dragonfly image
- Generate kernels to degrade high resolution image
- Characterize and subtract stellar halos in Dragonfly image

Installation
------------
- `python setup.py install` or `python setup.py develop` will do the job.
- `Python>=3` is needed, but you can try if `mrf` still works under `python2`.
- Check out the dependence of `mrf` depends from `requirements.txt`.

TODO
------------
- Analyse color terms between Dragonfly, CFHT, DES, HSC, etc.

Acknowledgement
---------------
Many scripts and snippets are from [`kungpao`](https://github.com/dr-guangtou/kungpao) written by [Song Huang](http://dr-guangtou.github.io) and Jiaxuan Li. I steal some functions here to make things more easily. Users will not be asked to install `kungpao` now.


License
-------
Copyright 2019 [Pieter van Dokkum](http://pietervandokkum.com) and [Jiaxuan Li](http://astrojacobli.github.io).

`mrf` is free software made available under MIT License. For details see the LICENSE file.


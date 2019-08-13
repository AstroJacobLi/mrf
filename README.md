# MRF: Multi-Resolution Filtering
Multi-Resolution Filtering: a method for isolating faint, extended emission in [Dragonfly](http://dragonflytelescope.org) data and other low resolution images.

<p align="center">
  <img src="df-logo.png" width="40%">
</p>

Applications
------------
- Download corresponding high resolution image (HSC, CFHT) of given Dragonfly image.
- Generate kernels to match high resolution image with low resolution one.
- Characterize and subtract stellar halos in Dragonfly image.
- Documentation is on its way :car: :airplane:

## Example

![MRF on NGC 5907](/Users/jiaxuanli/Research/Packages/mrf/demo.png)

Installation
------------

```bash
cd <install dir>
git clone git@github.com:AstroJacobLi/mrf.git
cd mrf
python setup.py install
```

If you don't have `git` configured, you can also download the `zip` file directly from https://github.com/AstroJacobLi/mrf/archive/master.zip, then unzip it and install in the same way. 

Then import `mrs` in Python:

```python
import mrf
print(mrf.__version__)
```

`Python>=3` is needed, but you can try whether `mrf` still works under `python2`. Check out the dependence of `mrf` depends from `requirements.txt`.

TODO
------------
- Analyze color terms between Dragonfly, CFHT, DES, HSC, etc.

Acknowledgement
---------------
Many scripts and snippets are from [`kungpao`](https://github.com/dr-guangtou/kungpao) (written by [Song Huang](http://dr-guangtou.github.io) and [Jiaxuan Li](http://astrojacobli.github.io)). Here we acknowledge @dr-guangtou for his help!


Citation
-------
If you use this code, please reference the `doi` below, and make sure to cite the dependencies as listed in [requirements](https://github.com/AstroJacobLi/mrf/blob/master/requirements.txt). 

`mrf` is a free software made available under MIT License. For details see the LICENSE file. Copyright 2019 [Pieter van Dokkum](http://pietervandokkum.com) and [Jiaxuan Li](http://astrojacobli.github.io). 
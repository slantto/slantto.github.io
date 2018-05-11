pnpnav
========

Python module that compares features extracted from aerial imagery to
a database of features extracted from reference imagery in order to compute
the pose of the airborne camera. Implements variations of bundle adjustment
/ Perspective-n-Point (PnP) algorithms.

## Requirements

This repository has some non-standard requirements. The major one is
the [GDAL][gdal] library, which you need python bindings for. Use of the
[Anaconda Python Distribution][conda] is recommended for easy dependency
resolution.

In addition, we currently use the ANT Center Navigation Utilities library, which
has some unique installation requirements. To install that, you'll need
a C++ compiler, cmake, and swig. After building with cmake, you need to add
the python bindings to your python path.

You also need the AFIT/AFRL neogeo_db package from antcenter.net

If you're using Ubuntu 14.04 you can get away with something like: 

```
sudo apt-get install build-essential cmake python-gdal libgdal-dev swig python-opencv
```

[conda]: https://store.continuum.io/cshop/anaconda/

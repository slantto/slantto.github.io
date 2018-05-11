# neogeo

neogeo is a framework for performing efficient large-scale aerial image
based navigation. neogeo implements a tightly-coupled grid filter that
uses a coarse-to-fine search strategy. neogeo will implement several
similarity metrics that may vary between accuracy, applicability, and
computational requirements

## Prerequisites

Before you can use neogeo_db, you'll need to install some
dependencies. neogeo uses GDAL, rasterio, mercantile, and opencv
so you'll need those libraries and their dependencies. Currently testing
packaging and deployment using the [Anaconda Python Distribution][conda]

[conda]: https://store.continuum.io/cshop/anaconda/

## License

neogeo is distributed probably under the DoD Community Source Agreement with
fallbacks to Apache or something like MIT/BSD/LOL

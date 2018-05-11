try:
    from setuptools import setup
except ImportError:
    from distutils.core import setup

config = {
    'description': 'neogeodb',
    'author': 'Donald Venable',
    'url': 'https://repository.antcenter.net/projects/NEOGEO/repos/neogeo_db/browse',
    'download_url': 'https://repository.antcenter.net/scm/neogeo/neogeo_db.git',
    'author_email': 'donald.venable@us.af.mil',
    'version': '0.1',
    'install_requires': ['nose', 'h5py', 'gdal', 'rasterio', 'mercantile', 'dask'],
    'packages': ['neogeodb'],
    'package_dir': {'': 'python'},
    'scripts': [],
    'name': 'neogeodb'
}

setup(**config)

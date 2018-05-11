try:
    from setuptools import setup
except ImportError:
    from distutils.core import setup

config = {
    'description': 'neogeo',
    'author': 'Donald Venable',
    'url': 'https://repository.antcenter.net/projects/NEOGEO/repos/neogeo/browse',
    'download_url': 'https://repository.antcenter.net/scm/neogeo/neogeo.git',
    'author_email': 'donald.venable@us.af.mil',
    'version': '0.1',
    'install_requires': ['nose', 'h5py', 'gdal', 'rasterio', 'mercantile'],
    'packages': ['neogeo'],
    'package_dir': {'': 'python'},
    'scripts': [],
    'name': 'neogeo'
}

setup(**config)

try:
    from setuptools import setup
except ImportError:
    from distutils.core import setup

config = {
    'description': 'PnP - Navigation pose estimates from vision',
    'author': 'Donald Venable',
    'url': 'https://repository.antcenter.net/projects/PNP/repos/pnpnav/browse',
    'download_url': 'https://venabled@repository.antcenter.net/scm/pnp/pnpnav.git',
    'author_email': 'donald.venable@us.af.mil',
    'version': '0.1',
    'install_requires': ['nose', 'tables', 'numpy', 'NavPy', 'neogeodb', 'mercantile'],
    'packages': ['pnpnav'],
    'name': 'pnpnav'
}

setup(**config)

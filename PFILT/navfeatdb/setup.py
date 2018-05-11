from distutils.core import setup

setup(
    name='navfeatdb',
    version='0.1',
    packages=['navfeatdb', 'navfeatdb/db', 'navfeatdb/ortho', 'navfeatdb/frames', 'navfeatdb/projection', 'navfeatdb/utils'],
    url='',
    license='',
    author='Donald Venable',
    author_email='donald.venable@us.af.mil',
    description='Python tools for constructing a Feature Database from airborne or satellite imagery',
    requires=['numpy', 'tables', 'simplekml', 'rasterio', 'bcolz', 'pandas']
)

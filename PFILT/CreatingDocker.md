The following steps were used:

0. I used the continuumio/anaconda base image
0. apt-get update
0. I had to apt-get install gcc and apt-get install g++ to get gdal to work with neogeodb
0. apt-get install libgl1-mesa-glx
0. conda install jpeg
0. conda install -c conda-forge opencv (must do this before gdal!)
0. conda install -c conda-forge gdal (if you do without the -c option, it will break when you go to run ParticleFilter way later...  Also, but doing it after opencv, it reverts opencv back to a 2.4.13 version)
0. conda install -c conda-forge pyflann

Note that all steps up to this point are stored in the image "wide_area_search_base".  So, you hopefully do not have to run the steps above.  Or, I should create a Dockerfile that does them... :)

## After image creation
After this, all of the code (ParticleFilter) plus all of the data are in a given directory.  I try to mount this directory into the docker container at /mounted as follows:
    docker run --name WAS_test -i -t -v c:/Users/heinerbk/Downloads/WideAreaSearch:/mounted wide_area_search_base /bin/bash

Unfortunately, the next couple of steps require the Internet, modify the image from the container, and also need the volume...

1. cd /mounted/neogeodb
1. python setup.py install
1. cd /mounted/neogeo
1. python setup.py install
1. cd /mounted/NavPy
1. python setup.py install
1. cd /mounted/pyPNP
1. python setup.py install

Now you can run it by 

1. cd /mounted/ParticleFilterVisNav
1. python ParticleFilter.py


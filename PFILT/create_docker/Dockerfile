##########
# Dockerfile to run the particle filter and vision aided navigation
##########

FROM continuumio/anaconda
LABEL author="Clark Taylor" email=clark.taylor.3@us.af.mil

#First, get the g++ compiler needed for some of the conda
# libraries we will install and a couple of other things that will
# be needed

RUN apt-get -y update \
    && apt-get -y install g++ libgl1-mesa-glx \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

#The order of conda stuff here is important.  OpenCV, when it 
#first runs will install a later version.  Then when you install 
#gdal, it knows it needs to link with opencv and will make it go 
#back to a prior version.  Then, everything works... :)
RUN conda install -y jpeg \
    && conda install -y -c conda-forge opencv \
    && conda install -y -c conda-forge gdal \
    && conda install -y -c conda-forge pyflann \
    && conda clean -y -a

#These next files assume that in the current context are directories
#containing the source for the following libraries: 
#neogeo, neogeodb, pyPNP, NavPy
#I am putting these in here rather than going to the repositories because:
# 1.  I want them all in one place (and no changes to affect them)
# 2.  Four of them (navfeatdb, neogeodb, neogeo, and pyPNP) are in the ANTcenter
#     repository, which requires access to "yet another system"
COPY . /py_libs
RUN cd /py_libs/neogeodb; \
    python setup.py install; \
    cd /py_libs/neogeo; \
    python setup.py install; \
    cd /py_libs/NavPy; \
    python setup.py install; \
    cd /py_libs/pyPNP; \
    python setup.py install; \
    cd /py_libs/navfeatdb; \
    python setup.py install

#Before running, set path = /opt/conda/bin:$PATH
#ENV PATH="/opt/conda/bin:${PATH}"
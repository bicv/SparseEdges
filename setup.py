#!/usr/bin/env python
# -*- coding: utf8 -*-

# from distutils.core import setup
from setuptools import setup

NAME = "SparseEdges"
version = "0.1"

setup(
    name = NAME,
    version = version,
    packages = [NAME],
    package_dir = {NAME: NAME},
#     exclude_package_data={NAME: ['database', 'mat', 'figures']},
    packages=find_packages(exclude=('database', 'mat', 'figures',)),
    include_package_data=True, 
   #         py_modules=["pp", "ppauto", "ppcommon", "pptransport", "ppworker"],
    author = "Laurent Perrinet INT - CNRS",
    author_email = "Laurent.Perrinet@univ-amu.fr",
    description = "SparseEdges: A bio-inspired sparse representation of edges in natural images.",
    long_description=open("README.md").read(),
    license = "GPLv2",
    keywords = ('computational neuroscience', 'simulation', 'analysis', 'visualization', 'computer vision'),
    url = 'https://github.com/meduz/' + NAME, # use the URL to the github repo
    download_url = 'https://github.com/meduz/' + NAME + '/tarball/' + version,
    classifiers = ['Development Status :: 3 - Alpha',
                   'Environment :: Console',
                   'License :: OSI Approved :: GNU General Public License (GPL)',
                   'Operating System :: POSIX',
                   'Topic :: Scientific/Engineering',
                   'Topic :: Utilities',
                  ],
     )

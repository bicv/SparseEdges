#!/usr/bin/env python

from distutils.core import setup

setup(
    name = "SparseEdges",
    version = "0.1",
    packages = ['SparseEdges'],
    package_dir = {'SparseEdges': ''},
    package_data={'SparseEdges': ['index.html', "README.md"]},
    author = "Laurent Perrinet INT - CNRS",
    author_email = "Laurent.Perrinet@univ-amu.fr",
    description = "SparseEdges is a collection of tools to support all tasks associated with the sparse representation of an image in a LogGabor pyramid.",
    long_description=open("README.md").read(),
    license = "GPLv2",
    keywords = ('computational neuroscience', 'simulation', 'analysis', 'visualization', 'parameters'),
    url = 'https://github.com/meduz/SparseEdges', # use the URL to the github repo
    download_url = 'https://github.com/meduz/SparseEdges/tarball/0.1',
    classifiers = ['Development Status :: 3 - Alpha',
                   'Environment :: Console',
                   'License :: OSI Approved :: GNU General Public License (GPL)',
                   'Operating System :: POSIX',
                   'Topic :: Scientific/Engineering',
                   'Topic :: Utilities',
                  ],
     )

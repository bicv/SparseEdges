#!/usr/bin/env python
# -*- coding: utf8 -*-

from setuptools import setup, find_packages

NAME = "SparseEdges"
version = "0.3"

setup(
    name = NAME,
    version = version,
    packages = find_packages(exclude=['contrib', 'docs', 'tests']),
     author = "Laurent Perrinet INT - CNRS",
    author_email = "Laurent.Perrinet@univ-amu.fr",
    description = "SparseEdges: A bio-inspired sparse representation of edges in natural images.",
    long_description=open("README.md").read(),
    license = "GPLv2",
    install_requires=['LogGabor'],
    keywords = ('computational neuroscience', 'simulation', 'analysis', 'visualization', 'biologically-inspired', 'computer vision'),
    url = 'https://github.com/bicv/' + NAME, # use the URL to the github repo
    download_url = 'https://github.com/bicv/' + NAME + '/tarball/' + version,
    classifiers = ['Development Status :: 3 - Alpha',
                   'Environment :: Console',
                   'License :: OSI Approved :: GNU General Public License (GPL)',
                   'Operating System :: POSIX',
                   'Topic :: Scientific/Engineering',
                   'Topic :: Utilities',
                   'Programming Language :: Python :: 2',
                   'Programming Language :: Python :: 2.7',
                   'Programming Language :: Python :: 3',
                   'Programming Language :: Python :: 3.5',
                  ],
     )

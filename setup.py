#!/usr/bin/env python
# -*- coding: utf8 -*-

from setuptools import setup, find_packages

NAME = "SparseEdges"
import SparseEdges
VERSION = SparseEdges.__version__ # << to change in __init__.py

setup(
    name = NAME,
    version = VERSION,
    packages = find_packages(exclude=['contrib', 'docs', 'probe']),
     author = "Laurent Perrinet INT - CNRS",
    author_email = "Laurent.Perrinet@univ-amu.fr",
    description = "SparseEdges: A bio-inspired sparse representation of edges in natural images.",
    # long_description=open("README.md", 'r', encoding='utf-8').read(),
    license = "GPLv2",
    install_requires=['LogGabor'],
    extras_require={
                'html' : [
                         'vispy',
                         'matplotlib'
                         'jupyter>=1.0']
    },
    keywords = ('computational neuroscience', 'simulation', 'analysis', 'visualization', 'biologically-inspired', 'computer vision'),
    url = 'https://github.com/bicv/' + NAME, # use the URL to the github repo
    download_url = 'https://github.com/bicv/' + NAME + '/tarball/' + VERSION,
    classifiers = ['Development Status :: 3 - Alpha',
                   'Environment :: Console',
                   'License :: OSI Approved :: GNU General Public License (GPL)',
                   'Operating System :: POSIX',
                   'Intended Audience :: Science/Research',
                   'Topic :: Scientific/Engineering :: Visualization',
                   'Topic :: Scientific/Engineering :: Medical Science Apps.',
                   'Topic :: Scientific/Engineering :: Image Recognition',
                   'Topic :: Scientific/Engineering',
                   'Topic :: Utilities',
                   'Programming Language :: Python :: 3',
                   'Programming Language :: Python :: 3.7',
                  ],
     )

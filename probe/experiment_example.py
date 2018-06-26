#! /usr/bin/env python
# -*- coding: utf8 -*-
from __future__ import division, print_function
"""

An example MP run.

To run:
$ python experiment_example.py 

To remove cache:
$ rm -fr **/example*

"""
__author__ = "(c) Laurent Perrinet INT - CNRS"


import numpy as np
from SparseEdges import SparseEdges
mp = SparseEdges('https://raw.githubusercontent.com/bicv/SparseEdges/master/default_param.py')
mp.N = 128

image = mp.imread('https://raw.githubusercontent.com/bicv/SparseEdges/master/database/lena256.png')

name = 'example'
white = mp.pipeline(image, do_whitening=True)

import os
matname = os.path.join(mp.pe.matpath, name + '.npy')
try:
    hack
    edges = np.load(matname)
except Exception:
    edges, C_res = mp.run_mp(white, verbose=True)
    np.save(matname, edges)    
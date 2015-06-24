#! /usr/bin/env python
# -*- coding: utf8 -*-
from __future__ import division, print_function
"""

$ python test/experiment_fig-sparselets.py ./figures

$ rm -fr **/SparseLets* **/**/SparseLets* 

"""
__author__ = "(c) Laurent Perrinet INT - CNRS"
    
from SparseEdges import SparseEdges
FORMATS = ['pdf', 'eps']
mps = []
sizes = [16, 32, 64, 128, 256]
N_image = 32
N = 1024

for size, size_str in zip(sizes, ['_016', '_032', '_064',  '_128', '']):
    mp = SparseEdges('https://raw.githubusercontent.com/meduz/SparseEdges/master/default_param.py')
    mp.pe.seed = 42
    mp.pe.datapath = '../../SLIP/database/'
    mp.set_size((size, size))
    downscale_factor = sizes[-1]/size # > 1
    mp.pe.N_image = int(N_image*downscale_factor)
    mp.pe.N = int(N/downscale_factor**2)
    mp.init()
    mp.process('SparseLets' + size_str)
    mps.append(mp)

import matplotlib.pyplot as plt
fig_width_pt = 600 # Get this from LaTeX using \showthe\columnwidth
inches_per_pt = 1.0/72.27               # Convert pt to inches
fig_width = fig_width_pt*inches_per_pt  # width in inches
fig = plt.figure(figsize=(fig_width, fig_width/1.618))

sizes = [16, 32, 64, 128, 256]
experiments = ['SparseLets_' + '%0.3d' % size for size in sizes]
experiments[-1] = 'SparseLets'
databases = ['serre07_distractors'] * len(experiments)
labels = [str(size) for size in sizes]
fig, ax, inset = mp.plot(fig=fig, mps=mps, experiments=experiments, databases=databases, 
                  labels=labels, scale=True)    
FORMATS = ['pdf', 'eps']
for ext in FORMATS: fig.savefig(mps[0].pe.figpath + 'SparseLets_B.' + ext)
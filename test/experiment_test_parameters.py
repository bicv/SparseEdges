#! /usr/bin/env python
# -*- coding: utf8 -*-
from __future__ import division, print_function
"""

Testing some parameters of the SparseEdges framework on its efficiency.

rm -fr mat/edges/testing_* mat/testing_* figures/edges/testing_* figures/testing_*
frioul_batch  -n "14,15,16"  -M 36 'python experiment_test_parameters.py'
frioul_batch -M 200 'python experiment_test_parameters.py'

"""
__author__ = "(c) Laurent Perrinet INT - CNRS"
import numpy as np
import matplotlib
matplotlib.use("Agg") # agg-backend, so we can create figures without x-server (no PDF, just PNG etc.)

from SparseEdges import SparseEdges
FORMATS = ['pdf', 'eps']

# TODO: here, we are more interested in the processing of the database, not the comparison - use the correct function
# TODO : annotate the efficiency of different LogGabor bases (RMSE?)
# TODO: make a circular mask to avoid border effects coming with whitening...

#! comparing databases
#!--------------------
mp = SparseEdges('default_param.py')
mp.N = 128
mp.pe.datapath = '/Users/lolo/pool/science/PerrinetBednar15/database/'
mp.process('testing_vanilla')
# TODO: CRF
mp.process('testing_noise', noise=mp.pe.noise)
mp.process('testing_vanilla', name_database='serre07_targets')

# TODO : make an experiment showing that using scale does not bring much
mps, experiments = [], []
v_alpha = np.linspace(0.3, 1., 9)
for MP_alpha in v_alpha:
    mp = SparseEdges('default_param.py')
    mp.N = 128
    mp.pe.datapath = '/Users/lolo/pool/science/PerrinetBednar15/database/'
    mp.pe.MP_alpha = MP_alpha
    exp = 'testing_MP_alpha_' + str(MP_alpha).replace('.', '_')
    mp.process(exp)
    experiments.append(exp)
    mps.append(mp)
    
import matplotlib.pyplot as plt
fig_width_pt = 318.670*.61 # Get this from LaTeX using \showthe\columnwidth
inches_per_pt = 1.0/72.27               # Convert pt to inches
fig_width = fig_width_pt*inches_per_pt  # width in inches

threshold = None
threshold = .25
databases = ['serre07_distractors'] * len(experiments)
labels = ['%0.2f' % MP_alpha for MP_alpha in v_alpha]
fig = plt.figure(figsize=(fig_width, fig_width/1.618))
fig, a, ax = plot(mps=mps,
                  experiments=experiments, databases=databases, labels=labels, 
                  fig=fig, color=[0., 1., 0.], threshold=threshold, scale=True)    
a.set_xlabel(r' $\alpha$')

mp.pe.figpath = 'test/figures'
import os
for ext in FORMATS: fig.savefig(os.path.join(mp.pe.figpath, 'testing_alpha.' + ext))
        
## TODO:  would be interesting to see how that changes with number of image patches used, i.e. whether it settles down to that particular pattern or just jumps around.
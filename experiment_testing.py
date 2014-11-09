#! /usr/bin/env python
# -*- coding: utf8 -*-
"""

Doing the real stuff: taking a bunch of images, computing edges and then doing
statistics on that.

rm -fr mat/edges/testing_* mat/testing_* figures/edges/testing_* figures/testing_*
frioul_batch  -n "14,15,16"  -M 36 'python experiment_testing.py'
frioul_batch -M 200 'python experiment_testing.py'

"""
__author__ = "(c) Laurent Perrinet INT - CNRS"
import matplotlib
matplotlib.use("Agg") # agg-backend, so we can create figures without x-server (no PDF, just PNG etc.)

import numpy as np
from NeuroTools.parameters import ParameterSet
from SLIP import Image
from LogGabor import LogGabor
from SparseEdges import SparseEdges, plot
from NeuroTools.parameters import ParameterSet
pe = ParameterSet('default_param.py')
FORMATS = ['pdf', 'eps']

def init_pe(pe, N_X=pe.N_X, N_image=pe.N_image, N=pe.N):
    pe.datapath = '../AssoField/database/'
    pe.N_image = N_image
    pe.N_X = N_X
    pe.N = N
    im = Image(pe)
    lg = LogGabor(im)
    mp = SparseEdges(lg)
    return mp

# TODO: here, we are more interested in the processing of the database, not the comparison - use the correct function
# TODO : annotate the efficiency of different LogGabor bases (RMSE?)
# TODO: make a circular mask to avoid border effects coming with whitening...

#! comparing databases
#!--------------------
mp = init_pe(pe)
mp.process('testing_vanilla')
mp.process('testing_noise', noise=pe.noise)
mp.process('testing_vanilla', name_database='serre07_targets')

pe = ParameterSet('default_param.py')

# TODO : make an experiment showing that using scale does not bring much
##! comparing representation parameters
##!------------------------------------
#! shorter log-gabor filters
pe_ = ParameterSet('default_param.py')# = pe.copy()
pe_.B_theta = pe.B_theta*2
mp = init_pe(pe_)
mp.process('testing_short')

#! longer log-gabor filters
pe_= ParameterSet('default_param.py')# = pe.copy()
pe_.B_theta = pe.B_theta/2
mp = init_pe(pe_)
mp.process('testing_long')

## other candidate parameters from class SparseEdges:
##   B_sf_ratio = 3.

import matplotlib.pyplot as plt
fig_width_pt = 318.670*.61 # Get this from LaTeX using \showthe\columnwidth
inches_per_pt = 1.0/72.27               # Convert pt to inches
fig_width = fig_width_pt*inches_per_pt  # width in inches

mps, experiments = [], []
v_alpha = np.linspace(0.3, 1., 5)
for MP_alpha in v_alpha:
    pe = ParameterSet('default_param.py')
    pe.MP_alpha = MP_alpha
    mp = init_pe(pe)
    exp = 'efficiency_MP_alpha_' + str(MP_alpha).replace('.', '_')
    mp.process(exp)
    experiments.append(exp)
    mps.append(mp)

threshold = .9
databases = ['serre07_distractors'] * len(experiments)
labels = ['%0.2f' % MP_alpha for MP_alpha in v_alpha]
fig = plt.figure(figsize=(fig_width, fig_width/1.618))
fig, a, ax = plot(mps=mps,
                  experiments=experiments, databases=databases, labels=labels, 
                  fig=fig, color=[0., 1., 0.], threshold=threshold, scale=True)    
a.set_xlabel(r' $\alpha$')
for ext in FORMATS: fig.savefig(mp.figpath + 'testing_alpha.' + ext)
        

## TODO:  would be interesting to see how that changes with number of image patches used, i.e. whether it settles down to that particular pattern or just jumps around.

#! /usr/bin/env python
# -*- coding: utf8 -*-
"""

Doing the real stuff: taking a bunch of images, computing edges and then doing
statistics on that.

rm -fr mat/edges/testing_* mat/testing_* figures/edges/testing_* figures/testing_*
frioul_batch  -n "14,15,16"  -M 36 'python experiment_testing.py'

"""
__author__ = "(c) Laurent Perrinet INT - CNRS"
import matplotlib
matplotlib.use("Agg") # agg-backend, so we can create figures without x-server (no PDF, just PNG etc.)

import numpy as np
from NeuroTools.parameters import ParameterSet
from SLIP import Image
from LogGabor import LogGabor
from SparseEdges import SparseEdges
from NeuroTools.parameters import ParameterSet
pe = ParameterSet('default_param.py')

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
for B_sf in np.logspace(-.5, .5, 5, base=10, endpoint=True)*pe.B_sf:
    pe.B_sf = B_sf
    mp = init_pe(pe)
    mp.process('testing_B_sf_' + str(B_sf).replace('.', '_'))

for B_theta in np.logspace(-.5, .5, 5, base=10, endpoint=True)*pe.B_theta:
    pe.B_theta = B_theta
    mp = init_pe(pe)
    mp.process('testing_B_theta_' + str(B_theta).replace('.', '_'))

for n_theta in [2, 3, 5, 8, 13, 21, 34]:
    pe.n_theta = n_theta
    mp = init_pe(pe)
    mp.process('testing_n_theta_' + str(n_theta).replace('.', '_'))

for base_levels in np.logspace(.25, 1.25, 15, base=2, endpoint=True):
    pe.base_levels = base_levels
    mp = init_pe(pe)
    mp.process('testing_base_levels_' + str(base_levels).replace('.', '_'))

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
##    n_levels = 5, n_theta = 16, B_sf_ratio = 3.
pe_ = ParameterSet('default_param.py')# = pe.copy()
pe_.n_theta = pe.n_theta*2
mp = init_pe(pe_)
mp.process('testing_moretheta')

# is whitening important?
pe_ = ParameterSet('default_param.py')
pe_.do_whitening = False
mp = init_pe(pe_)
# mp.process('testing_nowhite')

#! softy MP with a lower alpha value
pe_ = ParameterSet('default_param.py')
pe_.MP_alpha = .25
mp = init_pe(pe_)
mp.process('testing_smooth')

#! hard MP with a full alpha value
pe_ = ParameterSet('default_param.py')#
pe_.MP_alpha = 1.
mp = init_pe(pe_)
mp.process('testing_hard')

##! comparing edge methods
##!-----------------------
## parameters from class EdgeFactory:
pe_ = ParameterSet('default_param.py')#
pe_.N = pe.N*2
mp = init_pe(pe_)
mp.process('testing_moreN')

## TODO:  would be interesting to see how that changes with number of image patches used, i.e. whether it settles down to that particular pattern or just jumps around.

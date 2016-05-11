# -*- coding: utf8 -*-
from __future__ import division, print_function
"""

$ python experiment_retina_sparseness.py

rm -fr **/retina_sparseness* **/**/retina_sparseness*

"""
import numpy as np
from SparseEdges import SparseEdges
mps = []
for name_database in ['serre07_distractors']:#, 'serre07_distractors_urban', 'laboratory']:
    mp = SparseEdges('https://raw.githubusercontent.com/meduz/SparseEdges/master/default_param.py')
    mp.pe.datapath = '../../SLIP/database/'
    mp.pe.N_image = 100
    mp.pe.N = 2**14
    mp.pe.n_theta = 1
    mp.pe.B_theta = np.inf
    mp.pe.line_width = 0
    mp.init()
    # normal experiment
    imageslist, edgeslist, RMSE = mp.process(exp='retina_sparseness', name_database=name_database)
    mps.append(mp)
    # control experiment
    mp.pe.MP_alpha = np.inf
    mp.init()
    imageslist, edgeslist, RMSE = mp.process(exp='retina_sparseness_linear', name_database=name_database)
    mps.append(mp)
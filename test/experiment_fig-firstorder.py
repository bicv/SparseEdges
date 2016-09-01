# -*- coding: utf8 -*-
from __future__ import division, print_function
"""

$ python experiment_fig-firstorder.py

rm -fr **/prior_* **/**/prior_*
rm -fr **/prior_vanilla* **/**/prior_vanilla*
rm -fr **/prior_vanilla_noise_* **/**/prior_vanilla_noise_*
rm -fr **/prior_firstorder* **/**/prior_firstorder*

"""
import numpy as np
from SparseEdges import SparseEdges

for name_database in ['serre07_distractors']:#, 'serre07_distractors_urban', 'laboratory']:
    mp = SparseEdges('https://raw.githubusercontent.com/bicv/SparseEdges/master/default_param.py')
    mp.pe.datapath = '../../SLIP/database/'
    mp.pe.seed = 21341353 # this ensures that all image lists are the same for the different experiments
    mp.pe.N_image = 20
    mp.pe.N = 1024

    # control experiment
    #mp.theta = np.linspace(-np.pi/2, np.pi/2, mp.n_theta+1)[1:]
    imageslist, edgeslist, RMSE = mp.process(exp='prior_vanilla', name_database=name_database)

    imageslist_noise, edgeslist_noise, RMSE_noise = mp.process(exp='prior_vanilla_noise_' + str(mp.pe.noise).replace('.', '_'), name_database=name_database, noise=mp.pe.noise)

    if True:#try:
        six, N, N_image = edgeslist.shape
        # first-order prior
        v_hist, v_theta_edges = mp.histedges_theta(edgeslist, display=False)
        v_theta_middles, v_theta_bin  = (v_theta_edges[1:]+v_theta_edges[:-1])/2, v_theta_edges[1]-v_theta_edges[0]
        v_hist, v_theta_edges = mp.histedges_theta(edgeslist, display=False)
        v_theta_middles, v_theta_bin  = (v_theta_edges[1:]+v_theta_edges[:-1])/2, v_theta_edges[1]-v_theta_edges[0]

        z = np.linspace(0, 1., mp.pe.n_theta+1)
        P = np.cumsum(np.hstack((0, v_hist[-1]/2, v_hist[:-1], v_hist[-1]/2)))
        
        theta_prior = np.interp(z, P, np.hstack((v_theta_edges[0]-v_theta_bin/2, v_theta_edges[:-1], v_theta_edges[-1]-v_theta_bin/2)))

        mp = SparseEdges('https://raw.githubusercontent.com/bicv/SparseEdges/master/default_param.py')
        mp.pe.datapath = '../../SLIP/database/'
        mp.pe.seed = 21341353 # this ensures that all image lists are the same for the different experiments
        mp.pe.N_image = 20
        mp.pe.N = 1024
        
        imageslist, edgeslist, RMSE =  mp.process(exp='prior_firstorder', name_database=name_database)
        mp.MP_rho = .994304364466
        imageslist, edgeslist, RMSE = mp.process(exp='prior_quant_firstorder', name_database=name_database)
        mp.MP_rho = None
        imageslist_noise, edgeslist_noise, RMSE_noise = mp.process(exp='prior_firstorder_noise_' + str(mp.pe.noise).replace('.', '_'), name_database=name_database, noise=mp.pe.noise)
    #except:
    #    print('run again once first batches are finished ')

    mp.MP_rho = .994304364466
    imageslist, edgeslist, RMSE = mp.process(exp='prior_quant', name_database=name_database)
    mp.MP_rho = None
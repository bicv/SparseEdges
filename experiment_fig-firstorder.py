
"""

$ python experiment_fig-firstorder.py

rm -fr **/prior_vanilla* **/**/prior_vanilla*

"""
import __init__
import numpy as np
from NeuroTools.parameters import ParameterSet
from SLIP import Image
from LogGabor import LogGabor
from SparseEdges import SparseEdges

pe = ParameterSet('default_param.py')
pe.seed = 42 # this ensures that all image lists are the same for the different experiments
pe.N_image = 10
pe.N = 512
im = Image(pe)
lg = LogGabor(im)
mp = SparseEdges(lg)

for name_database in ['serre07_distractors']:#, 'laboratory']:
    # control experiment
    imageslist, edgeslist, RMSE = mp.process(exp='prior_vanilla', name_database=name_database)
    imageslist, edgeslist_noise, RMSE = mp.process(exp='prior_vanilla_noise', name_database=name_database, noise=pe.noise)

    try:
        # first-order prior
        v_hist, v_theta_edges = mp.histedges_theta(edgeslist, display=False)
        v_theta_middles, v_theta_bin  = (v_theta_edges[1:]+v_theta_edges[:-1])/2, v_theta_edges[1]-v_theta_edges[0]
        z = np.linspace(0, 1., pe.n_theta+2)
        P = np.cumsum(np.hstack((0, v_hist[-1]/2, v_hist[:-1], v_hist[-1]/2)))
        theta_prior = np.interp(z, P, np.hstack((v_theta_edges[-1]-np.pi, v_theta_edges))) #% np.pi
        mp.theta = (theta_prior[1:]) % (np.pi)
        
        imageslist, edgeslist, RMSE =  mp.process(exp='prior_vanilla_firstorder', name_database=name_database)
        imageslist, edgeslist, RMSE = mp.process(exp='prior_vanilla_firstorder_noise', name_database=name_database, noise=pe.noise)
    except:
        print('run again once first batches are finished ')
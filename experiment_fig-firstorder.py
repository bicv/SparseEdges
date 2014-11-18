
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
im = Image(pe)
lg = LogGabor(im)
mp = SparseEdges(lg)

pe.seed = 42 # this ensures that all image lists are the same for the different experiments

for name_database in ['serre07_distractors', 'laboratory']:
    # control experiment
    imageslist, edgeslist, RMSE = mp.process(exp='prior_vanilla', name_database=name_database)
    imageslist, edgeslist_noise, RMSE = mp.process(exp='prior_vanilla_noise', name_database=name_database, noise=pe.noise)

    try:
        # first-order prior
        v_hist, v_theta_edges = mp.histedges_theta(edgeslist, display=False)
        v_theta_middles, v_theta_bin  = (v_theta_edges[1:]+v_theta_edges[:-1])/2, v_theta_edges[1]-v_theta_edges[0]
        z = np.linspace(1./pe.n_theta, 1., pe.n_theta)
        mp.theta = np.interp(z, np.hstack((0, np.cumsum(v_hist))), (v_theta_edges-v_theta_bin/2))
        mp.theta = (mp.theta) % (np.pi)
        
        imageslist, edgeslist, RMSE =  mp.process(exp='prior_vanilla_firstorder', name_database=name_database)
        imageslist, edgeslist, RMSE = mp.process(exp='prior_vanilla_firstorder_noise', name_database=name_database, noise=pe.noise)
    except:
        print('run again once first batches are finished ')
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

name_database='serre07_distractors'
#name_database='laboratory'
y
# control experiment
imageslist, edgeslist, RMSE = mp.process(exp='prior_vanilla', name_database=name_database)
imageslist, edgeslist_noise, RMSE = mp.process(exp='prior_vanilla_noise', name_database=name_database, noise=pe.noise)

try:
    # first-order prior
    v_hist, v_theta_edges = mp.histedges_theta(edgeslist, display=False)
    z = np.linspace(0, 1.-1./pe.n_theta, pe.n_theta)

    mp.theta = np.interp(z, np.hstack((0, np.cumsum(v_hist))), (v_theta_edges-v_theta_bin/2))

    imageslist, edgeslist, RMSE =  mp.process(exp='prior_vanilla_firstorder', name_database=name_database)
    imageslist, edgeslist, RMSE = mp.process(exp='prior_vanilla_firstorder_noise', name_database=name_database, noise=pe.noise)
except:
    print('run again once first batches are finished ')
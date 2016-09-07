import numpy as np
from SparseEdges import SparseEdges

mp = SparseEdges('https://raw.githubusercontent.com/bicv/SparseEdges/master/default_param.py')
mp.N = 128 # number of edges
mp.pe.figsize_edges = 9

#! defining a reference test image (see test_Image)
image = np.zeros((mp.pe.N_X, mp.pe.N_Y))
image[mp.pe.N_X/2:mp.pe.N_X/2+mp.pe.N_X/4, mp.pe.N_X/2:mp.pe.N_X/2+mp.pe.N_X/4] = 1
image[mp.pe.N_X/2:mp.pe.N_X/2+mp.pe.N_X/4, mp.pe.N_X/4:mp.pe.N_X/2] = -1

import os
matname = os.path.join(mp.pe.matpath, 'experiment_test_MP.npy')
try:
    edges = np.load(matname)
except:
    edges, C_res = mp.run_mp(image, verbose=False)
    np.save(matname, edges)   
fig, a = mp.show_edges(edges, image=mp.whitening(image))
import numpy as np
from SparseEdges import SparseEdges

mp = SparseEdges('default_param.py')
mp.N = 128 # number of edges
mp.pe.figsize_edges = 9

#! defining a reference test image (see test_Image)
image = np.zeros((mp.N_X, mp.N_Y))
image[mp.N_X/2:mp.N_X/2+mp.N_X/4, mp.N_X/2:mp.N_X/2+mp.N_X/4] = 1
image[mp.N_X/2:mp.N_X/2+mp.N_X/4, mp.N_X/4:mp.N_X/2] = -1

matname = 'mat/experiment_test_MP.npy'
try:
    edges = np.load(matname)
except:
    edges, C_res = mp.run_mp(image, verbose=False)
    np.save(matname, edges)   
fig, a = mp.show_edges(edges, image=mp.whitening(image))
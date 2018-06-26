import numpy as np
from SparseEdges import SparseEdges
mp = SparseEdges('https://raw.githubusercontent.com/bicv/SparseEdges/master/default_param.py')
mp.pe.N = 2**9
mp.pe.MP_alpha = 1
mp.init()


image = mp.imread('https://raw.githubusercontent.com/bicv/SLIP/master/database/serre07_targets/B_N107001.jpg')

white = mp.pipeline(image, do_whitening=True)
    
import os
matname = os.path.join(mp.pe.matpath, 'experiment_test_whitening.npy')
try:
    edges = np.load(matname)
except Exception:
    edges, C_res = mp.run_mp(white, verbose=True)
    np.save(matname, edges)    


matname_MSE = os.path.join(mp.pe.matpath, 'experiment_test_whitening_MSE.npy')
try:
    MSE = np.load(matname_MSE)
except Exception:
    MSE = np.ones(mp.pe.N)
    image_rec = np.zeros_like(image)
    for i_N in range(mp.pe.N):
        MSE[i_N] =  ((white-image_rec)**2).sum()
        image_rec += mp.reconstruct(edges[:, i_N][:, np.newaxis])

    np.save(matname_MSE, MSE)     
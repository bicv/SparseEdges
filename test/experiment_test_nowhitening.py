import numpy as np
from SparseEdges import SparseEdges
mp = SparseEdges('default_param.py')
mp.N = 128
mp.pe.datapath = '/Users/lolo/pool/science/PerrinetBednar15/database/'
image = mp.imread(mp.datapath + 'serre07_targets/B_N107001.jpg')
mp.pe.figsize_edges = 9
image = mp.normalize(image, center=True)

#! trying now using no whitening of the image
mp.pe.do_whitening = False

matname = 'mat/experiment_test_nowhitening.npy'
matname_RMSE = 'mat/experiment_test_nowhitening_RMSE.npy'
try:
    edges = np.load(matname)
except:
    edges, C_res = mp.run_mp(image, verbose=True)
    np.save(matname, edges)    
    
fig, a = mp.show_edges(edges, image=image)
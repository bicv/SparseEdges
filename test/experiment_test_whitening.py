import numpy as np
from SparseEdges import SparseEdges
mp = SparseEdges('default_param.py')
mp.N = 128

image = mp.imread('/Users/lolo/pool/science/PerrinetBednar15/database/serre07_targets/B_N107001.jpg')
#print image.mean(), image.std()
image = mp.normalize(image, center=True)
#print image.mean(), image.std()

matname = 'mat/experiment_test_whitening.npy'
matname_RMSE = 'mat/experiment_test_whitening_RMSE.npy'
try:
    edges = np.load(matname)
    RMSE = np.load(matname_RMSE)
except:
    edges, C_res = mp.run_mp(image, verbose=True)
    np.save(matname, edges)    

    RMSE = np.ones(mp.N)
    image_ = image.copy()
    image_rec = np.zeros_like(image_)
    if mp.do_whitening: image_ = mp.whitening(image_)
    for i_N in range(mp.N):
        image_rec += mp.reconstruct(edges[:, i_N][:, np.newaxis])
        RMSE[i_N] =  ((image_*mp.mask-image_rec*mp.mask)**2).sum()

    np.save(matname_RMSE, RMSE)        
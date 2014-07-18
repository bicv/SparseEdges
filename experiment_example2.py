import __init__
import numpy as np
from NeuroTools.parameters import ParameterSet
from SLIP import Image
from LogGabor import LogGabor
from SparseEdges import SparseEdges
pe = ParameterSet('default_param.py')

# defining input image as Lena
from pylab import imread
image = imread('../AssoField/database/yelmo' + str(pe.N_X) + '.png')[:,:,0]#.flipud().fliplr()
print image.mean(), image.std()

pe.datapath = '../AssoField/database/'
pe.figsize_edges =12

im = Image(pe)
image = im.normalize(image, center=True)
print image.mean(), image.std()

lg = LogGabor(im)
mp = SparseEdges(lg)

matname = 'mat/example2.npy'
matname_RMSE = 'mat/example2_RMSE.npy'
try:
    edges = np.load(matname)
    RMSE = np.load(matname_RMSE)
except:
    edges, C_res = mp.run_mp(image, verbose=True)
    RMSE = np.ones(mp.N)
    image_ = image.copy()
    if mp.do_whitening: image_ = mp.im.whitening(image_)
    for i_N in range(mp.N):
        image_rec = mp.reconstruct(edges[:, :i_N])
        RMSE[i_N] =  ((image_-image_rec)**2).sum()

    np.save(matname, edges)    
    np.save(matname_RMSE, RMSE)        
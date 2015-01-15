import __init__
import numpy as np
from NeuroTools.parameters import ParameterSet
from SLIP import Image
from LogGabor import LogGabor
from SparseEdges import SparseEdges
pe = ParameterSet('default_param.py')

# defining input image as Lena
from pylab import imread
image = imread('database/yelmo' + str(pe.N_X) + '.png').mean(axis=-1)#.flipud().fliplr()
image = imread('database/lena' + str(pe.N_X) + '.png').mean(axis=-1)#.flipud().fliplr()
#print image.mean(), image.std()

pe.N = 512

im = Image(pe)
image = im.normalize(image, center=True)
#print image.mean(), image.std()
lg = LogGabor(im)
mp = SparseEdges(lg)

matname = 'mat/example.npy'
try:
    edges = np.load(matname)
except:
    edges, C_res = mp.run_mp(image, verbose=True)
    print edges.shape
    np.save(matname, edges)    

matname_RMSE = 'mat/example_RMSE.npy'
try:
    RMSE = np.load(matname_RMSE)
except:
    RMSE = np.ones(mp.N)
    image_ = image.copy()
    image_rec = np.zeros_like(image_)
    if mp.do_whitening: image_ = mp.im.whitening(image_)
    for i_N in range(mp.N):
        image_rec += mp.reconstruct(edges[:, i_N][:, np.newaxis])
        RMSE[i_N] =  ((image_*im.mask-image_rec*im.mask)**2).sum()

    np.save(matname_RMSE, RMSE)        
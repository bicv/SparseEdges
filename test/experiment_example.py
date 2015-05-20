
import numpy as np
from SLIP import Image
from LogGabor import LogGabor
from SparseEdges import SparseEdges
im = Image('default_param.py')

# defining input image as Lena
from pylab import imread
image = imread('database/yelmo' + str(im.N_X) + '.png').mean(axis=-1)
image = imread('database/lena' + str(im.N_X) + '.png').mean(axis=-1)
#print image.mean(), image.std()
#print pe.N_X

im.pe.N = 512
image = im.normalize(image, center=True)
#print image.mean(), image.std()
lg = LogGabor(im)
mp = SparseEdges(lg)

matname = 'mat/example.npy'
try:
    edges = np.load(matname)
except:
    edges, C_res = mp.run_mp(image, verbose=True)
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
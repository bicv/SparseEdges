import __init__
import numpy as np
from NeuroTools.parameters import ParameterSet
from SLIP import Image
from LogGabor import LogGabor
from SparseEdges import SparseEdges

# defining input image as Lena
from pylab import imread
#image = imread('../AssoField/database/yelmo256.png')[:,:,0]#.flipud().fliplr()
image = imread('../AssoField/database/yelmo64.png')[:,:,0]#.flipud().fliplr()
print image.mean(), image.std()

pe = ParameterSet('default_param.py')
pe.datapath = '../AssoField/database/'
pe.figsize_edges =12

#pe.N = 32 # number of edges

pe.N_X, pe.N_Y = image.shape
print pe
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
    edges, C_res, RMSE = mp.run_mp(image, verbose=True)
    np.save(matname, edges)    
    np.save(matname_RMSE, RMSE)        
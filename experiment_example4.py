import __init__
import numpy as np
from NeuroTools.parameters import ParameterSet
from SLIP import Image
from LogGabor import LogGabor
from SparseEdges import SparseEdges
pe = ParameterSet('default_param.py')
pe.N = 210
pe.B_sf = 1.5
#pe.do_whitening = False
#pe.base_levels = 4


# defining input image as Lena
from pylab import imread
image = imread('database/Geisler01Fig7A.png').mean(axis=-1)
print image.mean(), image.std()

im = Image(pe)
image = im.normalize(image, center=True)
print image.mean(), image.std()

lg = LogGabor(im)
mp = SparseEdges(lg)
print mp.n_levels, mp.sf_0

matname = 'mat/example4.npy'
try:
    edges = np.load(matname)
except:
    edges, C_res = mp.run_mp(image, verbose=True)
    np.save(matname, edges)    
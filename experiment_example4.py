import __init__
import numpy as np
from NeuroTools.parameters import ParameterSet
from SLIP import Image
from LogGabor import LogGabor
from SparseEdges import SparseEdges
pe = ParameterSet('default_param.py')
pe.N = 128

# defining input image as Lena
from pylab import imread
image = imread('database/Geisler01Fig7A.png').mean(axis=-1)
print image.mean(), image.std()

pe.figsize_edges = 12

im = Image(pe)
image = im.normalize(image, center=True)
print image.mean(), image.std()

lg = LogGabor(im)
mp = SparseEdges(lg)

matname = 'mat/example4.npy'
try:
    edges = np.load(matname)
except:
    edges, C_res = mp.run_mp(image, verbose=True)
    np.save(matname, edges)    
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
pe.figsize_edges = 12

im = Image(pe)
image = im.normalize(image, center=True)
print image.mean(), image.std()

lg = LogGabor(im)
mp = SparseEdges(lg)

#! trying now using no whitening of the image
pe.do_whitening=False
im = Image(pe)
lg = LogGabor(im)
mp = SparseEdges(lg)

matname = 'mat/example3.npy'
try:
    edges = np.load(matname)
except:
    edges, C_res = mp.run_mp(image, verbose=True)
    np.save(matname, edges)    
    
fig, a = mp.show_edges(edges, image=image)
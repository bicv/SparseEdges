
import matplotlib.pyplot as plt
import numpy as np
from NeuroTools.parameters import ParameterSet
from SLIP import Image
from LogGabor import LogGabor
from SparseEdges import SparseEdges

# defining input image as Lena
image = plt.imread('../AssoField/database/yelmo256.png')[:,:,0]#.flipud().fliplr()
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
try:
    edges = np.load(matname)
except:
    edges, C_res = mp.run_mp(image, verbose=True)
    np.save(matname, edges)    
    
fig, a = mp.show_edges(edges, image=image)
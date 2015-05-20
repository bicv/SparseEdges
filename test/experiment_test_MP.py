import numpy as np
from NeuroTools.parameters import ParameterSet
from SLIP import Image
from LogGabor import LogGabor
from SparseEdges import SparseEdges

pe = ParameterSet('default_param.py')
pe.N = 128 # number of edges
pe.figsize_edges = 9

#! defining a reference test image (see test_Image)
image = np.zeros((pe.N_X, pe.N_Y))
image[pe.N_X/2:pe.N_X/2+pe.N_X/4, pe.N_X/2:pe.N_X/2+pe.N_X/4] = 1
image[pe.N_X/2:pe.N_X/2+pe.N_X/4, pe.N_X/4:pe.N_X/2] = -1

im = Image(pe)
lg = LogGabor(im)
mp = SparseEdges(lg)
matname = 'mat/experiment_test_MP.npy'
try:
    edges = np.load(matname)
except:
    edges, C_res = mp.run_mp(image, verbose=False)
    np.save(matname, edges)   
fig, a = mp.show_edges(edges, image=im.whitening(image))
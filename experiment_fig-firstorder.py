
import numpy as np
from NeuroTools.parameters import ParameterSet
from SLIP import Image
from LogGabor import LogGabor
from SparseEdges import SparseEdges

pe = ParameterSet('default_param.py')
im = Image(pe)
lg = LogGabor(im)
mp = SparseEdges(lg)

imageslist, edgeslist, RMSE = mp.process(exp='testing_vanilla', name_database='serre07_distractors')

v_hist, v_theta_edges = mp.histedges_theta(edgeslist, display=False)

z = np.linspace(.5/pe.n_theta, 1.-.5/pe.n_theta, pe.n_theta)
mp.theta = np.interp(z, np.hstack((0, np.cumsum(v_hist))), v_theta_edges)

mp.process('testing_vanilla_firstorder')
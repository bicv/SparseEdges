import __init__
import numpy as np
from NeuroTools.parameters import ParameterSet
from SLIP import Image
from LogGabor import LogGabor
from SparseEdges import SparseEdges


def init_pe(pe):
    im = Image(pe)
    lg = LogGabor(im)
    mp = SparseEdges(lg)
    return mp

figpath = '../../BICV-book/BICV_INT/BICV-sparse/figures/'
experiments = []
v_B_sf = np.logspace(-.5, .5, 5, base=10, endpoint=True)*pe.B_sf
for B_sf in v_B_sf:
    pe = ParameterSet('default_param.py')
    pe.B_sf = B_sf
    mp = init_pe(pe)
    exp = 'testing_B_sf_' + str(B_sf).replace('.', '_')
    mp.process(exp)
    experiments.append(exp)

databases = ['serre07_distractors'] * len(experiments)
fig, a, ax = mp.plot(experiments=experiments, databases=databases, labels=labels)    
if not(figpath==None): fig.savefig(figpath + 'efficiency_A.pdf')
    
experiments = []
v_B_theta = np.logspace(-.5, .5, 5, base=10, endpoint=True)*pe.B_theta
for B_theta in v_B_theta:
    pe = ParameterSet('default_param.py')
    pe.B_theta = B_theta
    mp = init_pe(pe)
    exp = 'testing_B_theta_' + str(B_theta).replace('.', '_')
    mp.process(exp)
    experiments.append(exp)

databases = ['serre07_distractors'] * len(experiments)
fig, a, ax = mp.plot(experiments=experiments, databases=databases, labels=labels)    
if not(figpath==None): fig.savefig(figpath + 'efficiency_B.pdf')
    
experiments = []
v_n_theta = [2, 3, 5, 8, 13, 21, 34]
for n_theta in v_n_theta:
    pe = ParameterSet('default_param.py')
    pe.n_theta = n_theta
    mp = init_pe(pe)
    exp = 'testing_n_theta_' + str(n_theta).replace('.', '_')
    mp.process(exp)
    experiments.append(exp)

databases = ['serre07_distractors'] * len(experiments)
fig, a, ax = mp.plot(experiments=experiments, databases=databases, labels=labels)    
if not(figpath==None): fig.savefig(figpath + 'efficiency_C.pdf')
    

experiments = []
v_base_levels = np.logspace(.25, 1.25, 15, base=2, endpoint=True)
for base_levels in v_base_levels:
    pe = ParameterSet('default_param.py')
    pe.base_levels = base_levels
    mp = init_pe(pe)
    exp = 'testing_base_levels_' + str(base_levels).replace('.', '_')
    mp.process(exp)
    experiments.append(exp)

databases = ['serre07_distractors'] * len(experiments)
fig, a, ax = mp.plot(experiments=experiments, databases=databases, labels=labels)    
if not(figpath==None): fig.savefig(figpath + 'efficiency_D.pdf')

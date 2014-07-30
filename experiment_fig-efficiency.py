import __init__
import numpy as np
from NeuroTools.parameters import ParameterSet
from SLIP import Image
from LogGabor import LogGabor
from SparseEdges import SparseEdges
import sys
pe = ParameterSet('default_param.py')


def init_pe(pe, N_X=128, N_image=10, N=512):
    pe.seed = 123456
    pe.N_image = N_image
    pe.N_X = N_X
    pe.N = N
    im = Image(pe)
    lg = LogGabor(im)
    mp = SparseEdges(lg)
    return mp

try:
    figpath = sys.argv[1]
except:
    figpath = ''
    
experiments = []
v_B_sf = np.logspace(-.5, .5, 5, base=10, endpoint=True)*pe.B_sf
for B_sf in v_B_sf:
    pe = ParameterSet('default_param.py')
    pe.B_sf = B_sf
    mp = init_pe(pe)
    exp = 'efficiency_B_sf_' + str(B_sf).replace('.', '_')
    mp.process(exp)
    experiments.append(exp)

databases = ['serre07_distractors'] * len(experiments)
labels = [str(B_sf) for B_sf in v_B_sf]
fig, a, ax = mp.plot(experiments=experiments, databases=databases, labels=labels)    
if not(figpath==''): fig.savefig(figpath + 'efficiency_A.pdf')
    
experiments = []
v_B_theta = np.logspace(-.5, .5, 5, base=10, endpoint=True)*pe.B_theta
for B_theta in v_B_theta:
    pe = ParameterSet('default_param.py')
    pe.B_theta = B_theta
    mp = init_pe(pe)
    exp = 'efficiency_B_theta_' + str(B_theta).replace('.', '_')
    mp.process(exp)
    experiments.append(exp)

databases = ['serre07_distractors'] * len(experiments)
labels = [str(B_theta) for B_theta in v_B_theta]
fig, a, ax = mp.plot(experiments=experiments, databases=databases, labels=labels)    
if not(figpath==''): fig.savefig(figpath + 'efficiency_B.pdf')
    
experiments = []
v_n_theta = [4, 6, 12, 24, 48]
for n_theta in v_n_theta:
    pe = ParameterSet('default_param.py')
    pe.n_theta = n_theta
    mp = init_pe(pe, N_image=10)
    exp = 'efficiency_n_theta_' + str(n_theta).replace('.', '_')
    mp.process(exp)
    experiments.append(exp)

databases = ['serre07_distractors'] * len(experiments)
labels = [str(n_theta) for n_theta in v_n_theta]
fig, a, ax = mp.plot(experiments=experiments, databases=databases, labels=labels)    
if not(figpath==''): fig.savefig(figpath + 'efficiency_C.pdf')
    
experiments = []
v_base_levels = [np.sqrt(2), np.sqrt(5)/2.+.5, np.sqrt(3), 2. , np.sqrt(5)]
#np.logspace(.25, 1.25, 5, base=2, endpoint=True)
for base_levels in v_base_levels:
    pe = ParameterSet('default_param.py')
    pe.base_levels = base_levels
    mp = init_pe(pe, N_image=10)
    exp = 'efficiency_base_levels_' + str(base_levels).replace('.', '_')
    mp.process(exp)
    experiments.append(exp)

databases = ['serre07_distractors'] * len(experiments)
labels = [str(base_levels) for base_levels in v_base_levels]
fig, a, ax = mp.plot(experiments=experiments, databases=databases, labels=labels)    
if not(figpath==''): fig.savefig(figpath + 'efficiency_D.pdf')
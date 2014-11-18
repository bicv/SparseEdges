"""

$ python experiment_fig-efficiency.py ../../CNRS/BICV-book/BICV_INT/BICV-sparse/

rm -fr **/efficiency_* **/**/efficiency_* 

"""

import __init__
import numpy as np
from NeuroTools.parameters import ParameterSet
from SLIP import Image
from LogGabor import LogGabor
from SparseEdges import SparseEdges, plot
import sys
pe = ParameterSet('default_param.py')
FORMATS = ['pdf', 'eps']

threshold = None # classical plots
threshold = .5 # plot sparseness obtained when reaching this threshold

def init_pe(pe, N_image=10, N=512):
    pe.seed = 123456
    pe.N_image = N_image
    #pe.N_X = N_X
    pe.N = N
    im = Image(pe)
    lg = LogGabor(im)
    mp = SparseEdges(lg)
    return mp

dofig = True
try:
    figpath = sys.argv[1]
except:
    dofig = False

import matplotlib.pyplot as plt
fig_width_pt = 318.670*.61 # Get this from LaTeX using \showthe\columnwidth
inches_per_pt = 1.0/72.27               # Convert pt to inches
fig_width = fig_width_pt*inches_per_pt  # width in inches

mps, experiments = [], []
v_B_sf = np.logspace(-.2, .2, 5, base=10, endpoint=True)*pe.B_sf
for B_sf in v_B_sf:
    pe = ParameterSet('default_param.py')
    pe.B_sf = B_sf
    mp = init_pe(pe)
    exp = 'efficiency_B_sf_' + str(B_sf).replace('.', '_')
    mp.process(exp)
    experiments.append(exp)
    mps.append(mp)

databases = ['serre07_distractors'] * len(experiments)
labels = ['%0.2f' % B_sf for B_sf in v_B_sf]
fig = plt.figure(figsize=(fig_width, fig_width/1.618))
fig, a, ax = plot(mps=mps,
                  experiments=experiments, databases=databases, labels=labels, 
                  fig=fig, color=[0., 1., 0.], threshold=threshold, scale=True)    
a.set_xlabel(r'frequency bandwith $B_{sf}$')
if dofig: 
    for ext in FORMATS: fig.savefig(figpath + 'efficiency_A.' + ext)
    
mps, experiments = [], []
v_B_theta = np.logspace(-.5, .5, 5, base=10, endpoint=True)*pe.B_theta
for B_theta in v_B_theta:
    pe = ParameterSet('default_param.py')
    pe.B_theta = B_theta
    mp = init_pe(pe)
    exp = 'efficiency_B_theta_' + str(B_theta).replace('.', '_')
    mp.process(exp)
    experiments.append(exp)
    mps.append(mp)

databases = ['serre07_distractors'] * len(experiments)
labels = ['%0.2f' % B_theta for B_theta in v_B_theta]
fig = plt.figure(figsize=(fig_width, fig_width/1.618))
fig, a, ax = plot(mps=mps, 
                  experiments=experiments, databases=databases, labels=labels, 
                  fig=fig, threshold=threshold, scale=True, color=[0., 1., 0.])    
a.set_xlabel(r'orientation bandwith $B_{\theta}$ (radians)')
if dofig: 
    for ext in FORMATS: fig.savefig(figpath + 'efficiency_B.' + ext)
    
mps, experiments = [], []
v_n_theta = [6, 12, 24, 48]
for n_theta in v_n_theta:
    pe = ParameterSet('default_param.py')
    pe.n_theta = n_theta
    mp = init_pe(pe)
    exp = 'efficiency_n_theta_' + str(n_theta).replace('.', '_')
    mp.process(exp)
    experiments.append(exp)
    mps.append(mp)

databases = ['serre07_distractors'] * len(experiments)
labels = [str(n_theta) for n_theta in v_n_theta]
fig = plt.figure(figsize=(fig_width, fig_width/1.618))
fig, a, ax = plot(mps=mps, 
                  experiments=experiments, databases=databases, labels=labels, 
                  fig=fig, threshold=threshold, scale=True, color=[0., 1., 0.])    
a.set_xlabel(r'number of orientations $N_{\theta}$')
if dofig: 
    for ext in FORMATS: fig.savefig(figpath + 'efficiency_C.' + ext)
    
mps, experiments = [], []
v_base_levels = [np.sqrt(2), np.sqrt(5)/2.+.5, np.sqrt(3), 2. , np.sqrt(5)]
#np.logspace(.25, 1.25, 5, base=2, endpoint=True)
for base_levels in v_base_levels:
    pe = ParameterSet('default_param.py')
    pe.base_levels = base_levels
    mp = init_pe(pe)
    exp = 'efficiency_base_levels_' + str(base_levels).replace('.', '_')
    mp.process(exp)
    experiments.append(exp)
    mps.append(mp)

databases = ['serre07_distractors'] * len(experiments)
labels = ['%0.2f' % (base_levels) for base_levels in v_base_levels]
labels[0] = r'$\sqrt{2}$'
labels[1] = r'$\phi$'
fig = plt.figure(figsize=(fig_width, fig_width/1.618))
fig, a, ax = plot(mps=mps, 
                  experiments=experiments, databases=databases, labels=labels, 
                  fig=fig, threshold=threshold, scale=True, color=[0., 1., 0.])    
a.set_xlabel(r'scale ratio')
if dofig: 
    for ext in FORMATS: fig.savefig(figpath + 'efficiency_D.' + ext)
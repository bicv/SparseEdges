#! /usr/bin/env python
# -*- coding: utf8 -*-
from __future__ import division, print_function
"""

$ python experiment_fig-efficiency.py ./figures

rm -fr **/efficiency_* **/**/efficiency_* 

"""
import numpy as np
from SparseEdges import SparseEdges
FORMATS = ['pdf', 'eps']

threshold = None # classical plots
threshold = .1 # plot L0 sparseness obtained when reaching this threshold
 
mp = SparseEdges('https://raw.githubusercontent.com/meduz/SparseEdges/master/default_param.py')
def init_mp():
    mp = SparseEdges('https://raw.githubusercontent.com/meduz/SparseEdges/master/default_param.py')
    mp.pe.seed = 42
    mp.pe.N_image = 20
    mp.pe.datapath = '../../SLIP/database/'
    return mp

    
FORMATS = ['pdf', 'eps']
#FORMATS = ['png']
import matplotlib
matplotlib.use('Agg') 
#matplotlib.rcParams.update({'font.size': 18, 'font.family': 'STIXGeneral', 'mathtext.fontset': 'stix'})
matplotlib.rcParams.update({'text.usetex': False})

import matplotlib.pyplot as plt
fig_width_pt = 800 #318.67085 # Get this from LaTeX using \showthe\columnwidth
inches_per_pt = 1.0/72.27               # Convert pt to inches
fig_width = fig_width_pt*inches_per_pt  # width in inches

# ==================================================================================================#
fig, [[A, B], [C, D]] = plt.subplots(2, 2, figsize=(fig_width, fig_width), subplot_kw={'axisbg':'w'})
# ==================================================================================================#
mps, experiments = [], []
v_B_sf = np.logspace(-.2, .2, 5, base=10, endpoint=True)*mp.pe.B_sf
for B_sf in v_B_sf:
    mp = init_mp()
    mp.pe.B_sf = B_sf
    exp = 'efficiency_B_sf_' + str(B_sf).replace('.', '_')
    mp.process(exp)
    experiments.append(exp)
    mps.append(mp)

databases = ['serre07_distractors'] * len(experiments)
labels = ['%0.2f' % B_sf for B_sf in v_B_sf]
try:
    fig, A, inset = mp.plot(mps=mps,
                      experiments=experiments, databases=databases, labels=labels, ref=2,
                      fig=fig, ax=A, color=[0., 1., 0.], threshold=threshold, scale=False)    
    A.set_xlabel(r'frequency bandwith $B_{sf}$')
    #A.set_yticks([0., 0.02, 0.04, 0.06])
except Exception as e:
    print('Failed to plot  with error : %s ' % e )

# ==================================================================================================#    
mps, experiments = [], []
v_B_theta = np.logspace(-.5, .5, 5, base=10, endpoint=True)*mp.pe.B_theta
for B_theta in v_B_theta:
    mp = init_mp()
    mp.pe.B_theta = B_theta
    exp = 'efficiency_B_theta_' + str(B_theta).replace('.', '_')
    mp.process(exp)
    experiments.append(exp)
    mps.append(mp)

databases = ['serre07_distractors'] * len(experiments)
labels = ['%0.2f' % B_theta for B_theta in v_B_theta]
try:
    fig, B, inset = mp.plot(mps=mps, 
                      experiments=experiments, databases=databases, labels=labels, ref=2, 
                      fig=fig, ax=B, threshold=threshold, scale=False, color=[0., 1., 0.])    
    B.set_xlabel(r'orientation bandwith $B_{\theta}$ (radians)')
    B.set_ylabel('')
    #B.set_yticks([0., 0.02, 0.04, 0.06])
    #B.set_yticklabels(['', '', '', ''])
except Exception as e:
    print('Failed to plot  with error : %s ' % e )

# ==================================================================================================#    
mps, experiments = [], []
v_n_theta = [6, 12, 24, 48]
for n_theta in v_n_theta:
    mp = init_mp()
    mp.pe.n_theta = n_theta
    mp = init_mp()
    exp = 'efficiency_n_theta_' + str(n_theta).replace('.', '_')
    mp.process(exp)
    experiments.append(exp)
    mps.append(mp)

databases = ['serre07_distractors'] * len(experiments)
labels = [str(n_theta) for n_theta in v_n_theta]
try:
    fig, C, inset = mp.plot(mps=mps, 
                      experiments=experiments, databases=databases, labels=labels, ref=2, 
                      fig=fig, ax=C, threshold=threshold, scale=True, color=[0., 1., 0.])    
    C.set_xlabel(r'number of orientations $N_{\theta}$')
    #C.set_yticks([0., 0.02, 0.04, 0.06])
except Exception as e:
    print('Failed to plot  with error : %s ' % e )

# ==================================================================================================#    
mps, experiments = [], []
v_base_levels = [np.sqrt(2), np.sqrt(5)/2.+.5, np.sqrt(3), 2. , np.sqrt(5)]
#np.logspace(.25, 1.25, 5, base=2, endpoint=True)
for base_levels in v_base_levels:
    mp = init_mp()
    mp.pe.base_levels = base_levels
    mp = init_mp()
    exp = 'efficiency_base_levels_' + str(base_levels).replace('.', '_')
    mp.process(exp)
    experiments.append(exp)
    mps.append(mp)

databases = ['serre07_distractors'] * len(experiments)
labels = ['%0.2f' % (base_levels) for base_levels in v_base_levels]
labels[0] = r'$\sqrt{2}$'
labels[1] = r'$\phi$'
labels[3] = '2'
try:
    fig, D, inset = mp.plot(mps=mps, 
                      experiments=experiments, databases=databases, labels=labels, ref=3, 
                      fig=fig, ax=D, threshold=threshold, scale=True, color=[0., 1., 0.])    
    D.set_xlabel(r'scale ratio')
    D.set_ylabel('')
    D.set_yticks([0., 1., 1.3])
    D.set_yticklabels(['0', '1', ''])
except Exception as e:
    print('Failed to plot  with error : %s ' % e )

for ax, label in zip([A, B, C, D], ['A', 'B', 'C', 'D']):
    ax.text(-.1, .95, label, transform=ax.transAxes, fontsize=12) #'('+label+')'
    ax.set_ylim([0., 1.6])
    ax.set_yticks([0., 1., 1.4])
    ax.set_yticklabels(["0", '1', ''])
    if label in ['B', 'D']: ax.set_yticklabels(['', '', ''])


# TODO : show CRF
        
#The parameter meanings (and suggested defaults) are::
#
#  left  = 0.125  # the left side of the subplots of the figure
#  right = 0.9    # the right side of the subplots of the figure
#  bottom = 0.1   # the bottom of the subplots of the figure
#  top = 0.9      # the top of the subplots of the figure
#  wspace = 0.2   # the amount of width reserved for blank space between subplots
#  hspace = 0.2   # the amount of height reserved for white space between subplots
fig.subplots_adjust(wspace=0.12, hspace=0.3,
                            left=0.125, right=0.98,
                            top=0.98,    bottom=0.12)
    
for ext in FORMATS: fig.savefig(mp.pe.figpath + 'efficiency.' + ext)
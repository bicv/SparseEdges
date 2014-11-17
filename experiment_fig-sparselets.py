"""

$ python experiment_fig-sparselets.py ../../CNRS/BICV-book/BICV_INT/BICV-sparse/

$ rm -fr **/SparseLets* **/**/SparseLets* 

"""
import __init__
from NeuroTools.parameters import ParameterSet
from SLIP import Image
from LogGabor import LogGabor
from SparseEdges import SparseEdges, plot
import sys

pe = ParameterSet('default_param.py')

def init_pe(pe, N_X=pe.N_X, N_image=pe.N_image, N=pe.N):
    pe.N_image = N_image
    pe.N_X = N_X
    pe.N_Y = N_X
    pe.N = N
    pe.seed = 42
    im = Image(pe)
    lg = LogGabor(im)
    mp = SparseEdges(lg)
    return mp

dofig = True
try:
    figpath = sys.argv[1]
except:
    dofig = False
    
mps = []
for size, size_str in zip([16, 32, 64, 128, 256], ['_016', '_032', '_064',  '_128', '']):
    mp = init_pe(pe, N_X=size, N_image=pe.N_image*pe.N_X/size, N=pe.N*size**2/pe.N_X**2)
    mp.process('SparseLets' + size_str)
    mps.append(mp)

import matplotlib.pyplot as plt
fig_width_pt = 318.670 # Get this from LaTeX using \showthe\columnwidth
inches_per_pt = 1.0/72.27               # Convert pt to inches
fig_width = fig_width_pt*inches_per_pt  # width in inches
fig = plt.figure(figsize=(fig_width, fig_width/1.618))

sizes = [16, 32, 64, 128, 256]
experiments = ['SparseLets_' + '%0.3d' % size for size in sizes] # ['testing_vanilla_016', 'testing_vanilla_032', 'testing_vanilla_064', 'testing_vanilla_128', 'testing_vanilla']
experiments[-1] = 'SparseLets'
databases = ['serre07_distractors'] * len(experiments)
labels = [str(size) for size in sizes]
fig, a, ax = plot(fig=fig, mps=mps, experiments=experiments, databases=databases, 
                  labels=labels, scale=True)    
if dofig: 
    FORMATS = ['pdf', 'eps']
    for ext in FORMATS: fig.savefig(figpath + 'SparseLets_B.' + ext)
"""

$ python test/experiment_fig-sparselets.py ../../CNRS/BICV-book/BICV_sparse/src/

$ rm -fr **/SparseLets* **/**/SparseLets* 

"""

__author__ = "(c) Laurent Perrinet INT - CNRS"
import matplotlib
matplotlib.use("Agg") # agg-backend, so we can create figures without x-server (no PDF, just PNG etc.)

dofig = True
try:
    import sys
    path = sys.argv[1]
except:
    path = ''
    dofig = False

    
from SparseEdges import SparseEdges
FORMATS = ['pdf', 'eps']
mps = []
sizes = [16, 32, 64, 128, 256]
N_image = 32
N = 1024

for size, size_str in zip(sizes, ['_016', '_032', '_064',  '_128', '']):
    mp = SparseEdges('https://raw.githubusercontent.com/meduz/SparseEdges/master/default_param.py')
    mp.pe.seed = 42
    mp.N = 128
    mp.pe.datapath = '/Users/lolo/pool/science/PerrinetBednar15/database/'
    mp.set_size((size, size))
    mp.pe.N_image= int(N_image*sizes[-1]/size)
    mp.N = int(N*(size/sizes[-1])**2)
    mp.process('SparseLets' + size_str)
    mps.append(mp)

    
import matplotlib.pyplot as plt
fig_width_pt = 318.670 # Get this from LaTeX using \showthe\columnwidth
inches_per_pt = 1.0/72.27               # Convert pt to inches
fig_width = fig_width_pt*inches_per_pt  # width in inches
fig = plt.figure(figsize=(fig_width, fig_width/1.618))

sizes = [16, 32, 64, 128, 256]
experiments = ['SparseLets_' + '%0.3d' % size for size in sizes]
experiments[-1] = 'SparseLets'
databases = ['serre07_distractors'] * len(experiments)
labels = [str(size) for size in sizes]
fig, a, ax = mp.plot(fig=fig, mps=mps, experiments=experiments, databases=databases, 
                  labels=labels, scale=True)    
if dofig: 
    FORMATS = ['pdf', 'eps']
    for ext in FORMATS: fig.savefig(mps[0].pe.figpath + 'SparseLets_B.' + ext)
else:
    fig.show()
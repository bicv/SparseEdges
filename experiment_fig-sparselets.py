import __init__
from NeuroTools.parameters import ParameterSet
from SLIP import Image
from LogGabor import LogGabor
from SparseEdges import SparseEdges, plot

pe = ParameterSet('default_param.py')

def init_pe(pe, N_X=pe.N_X, N_image=pe.N_image, N=pe.N):
    pe.N_image = N_image
    pe.N_X = N_X
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
    
mps = []
for size, size_str in zip([16, 32, 64, 128, 256], ['_016', '_032', '_064',  '_128', '']):
    mp = init_pe(pe, N_X=size, N_image=pe.N_image*pe.N_X/size, N=pe.N*size**2/pe.N_X**2)
    mp.process('SparseLets' + size_str)
    mps.append(mp)

sizes = [16, 32, 64, 128, 256]
experiments = ['SparseLets_' + '%0.3d' % size for size in sizes] # ['testing_vanilla_016', 'testing_vanilla_032', 'testing_vanilla_064', 'testing_vanilla_128', 'testing_vanilla']
experiments[-1] = 'SparseLets'
databases = ['serre07_distractors'] * len(experiments)
labels = [str(size) for size in sizes]
fig, a, ax = plot(mps=mps, experiments=experiments, databases=databases, labels=labels, scale=False)    
if dofig: fig.savefig(figpath + 'SparseLets_B.pdf')
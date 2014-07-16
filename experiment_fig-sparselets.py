
from NeuroTools.parameters import ParameterSet
from SLIP import Image
from LogGabor import LogGabor
from SparseEdges import SparseEdges

pe = ParameterSet('default_param.py')

def init_pe(pe, N_X=256, N_image=40, N=2048):
    pe.datapath = '../AssoField/database/'
    pe.N_image = N_image
    pe.N_X = N_X
    pe.N = N
    im = Image(pe)
    lg = LogGabor(im)
    mp = SparseEdges(lg)
    return mp

for size, size_str in zip([16, 32, 64, 128, 256], ['_016', '_064', '_032', '_128', '']):
    mp = init_pe(pe, N_X=size, N_image=40*256/size, N=2048*size**2/256**2)
    mp.process('testing_vanilla' + size_str)
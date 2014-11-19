import __init__
import numpy as np
from NeuroTools.parameters import ParameterSet
from SLIP import Image
from LogGabor import LogGabor
from SparseEdges import SparseEdges
pe = ParameterSet('default_param.py')
pe.N = 210
pe.N = 64
#pe.B_sf = 1.5
#pe.do_whitening = False
#pe.base_levels = 4
pe.n_theta = 48
pe.do_whitening = True

# defining input image as Lena
from pylab import imread
image = imread('database/Geisler01Fig7A.png').mean(axis=-1)
print image.mean(), image.std()

im = Image(pe)
image = im.normalize(image, center=True)
print image.mean(), image.std()

lg = LogGabor(im)
mp = SparseEdges(lg)
print mp.n_levels, mp.sf_0

print ' without second-order '
matname = 'mat/Geisler01Fig7A.npy'
try:
    edges = np.load(matname)
except:
    mp.pe.eta_SO = 0.
    edges, C_res = mp.run_mp(image, verbose=True)
    np.save(matname, edges)    

print ' with second-order '
matname = 'mat/Geisler01Fig7A_secondorder.npy'
try:
    edges = np.load(matname)
except:
    mp.pe.eta_SO = .75
    edges, C_res = mp.run_mp(image, verbose=True)
    np.save(matname, edges)    

    
for mp.pe.eta_SO in np.linspace(.5, 2.5, 9):
    edges, C_res = mp.run_mp(image, verbose=True)
    fig, a = mp.show_edges(edges, image=image, v_min=v_min, v_max=v_max, color='toto', show_phase=False) #
    fig.savefig(mp.pe.figpath + 'Geisler01Fig7_A.' + str(mp.pe.eta_SO) + '.pdf')
    